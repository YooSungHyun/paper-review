"""
PEAR SFT Trainer - Precomputed Behavior Probabilities 버전

미리 계산된 behavior policy (πβ) 확률값을 사용하여
reference model을 메모리에 올리지 않고 PEAR를 학습합니다.

사전 준비:
    1. compute_behavior_probs.py로 behavior_log_probs 계산
    2. 데이터셋에 behavior_log_probs 컬럼 추가

사용법:
    from for_rl_sft_trainer_precomputed import PEARSFTTrainerPrecomputed
    
    trainer = PEARSFTTrainerPrecomputed(
        model="Qwen/Qwen2.5-1.5B",
        train_dataset=dataset_with_behavior_probs,  # behavior_log_probs 포함
        weighting_mode="suffix",
    )
    trainer.train()

논문: Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning
arXiv: 2602.01058v1 [cs.LG] 1 Feb 2026
"""

import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig


class PEARSFTTrainerPrecomputed(SFTTrainer):
    """
    Precomputed behavior probabilities를 사용하는 PEAR Trainer.
    
    Reference model을 메모리에 올리지 않고, 미리 계산된 πβ 확률값을 사용합니다.
    
    Args:
        model: 학습할 모델 (target policy πθ)
        train_dataset: behavior_log_probs 컬럼을 포함하는 데이터셋
        weighting_mode: Importance weighting 방식
            - "uniform": Sequence-level weighting
            - "suffix": Token-level suffix-based weighting (default)
            - "block": Block-level weighting
        block_size: Block-level mode의 블록 크기 (default: 1)
        gamma: Discount factor (default: 0.999)
        clip_ratio_range: Per-token log-ratio clipping [lower, upper]
        clip_weight_range: Final weight clipping [min, max]
    
    Examples:
        >>> from for_rl_sft_trainer_precomputed import PEARSFTTrainerPrecomputed
        >>> from datasets import load_dataset
        >>> 
        >>> # behavior_log_probs가 포함된 데이터셋 로드
        >>> dataset = load_dataset("json", data_files="train_with_probs.jsonl")
        >>> 
        >>> trainer = PEARSFTTrainerPrecomputed(
        ...     model="Qwen/Qwen2.5-1.5B",
        ...     train_dataset=dataset["train"],
        ...     weighting_mode="suffix",
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: "str | PreTrainedModel",
        args: SFTConfig | TrainingArguments | None = None,
        data_collator: Optional[Any] = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional[Any] = None,
        formatting_func: Callable[[dict], str] | None = None,
        # PEAR 전용 파라미터
        weighting_mode: str = "suffix",
        block_size: int = 1,
        gamma: float = 0.999,
        clip_ratio_range: tuple[float, float] = (-0.08, 0.3),
        clip_weight_range: tuple[float, float] = (0.1, 10.0),
    ):
        """PEAR SFT Trainer (Precomputed) 초기화."""
        
        # PEAR 파라미터 저장
        self.weighting_mode = weighting_mode
        self.block_size = block_size
        self.gamma = gamma
        self.clip_ratio_range = clip_ratio_range
        self.clip_weight_range = clip_weight_range
        
        # 파라미터 검증
        self._validate_pear_params()
        
        # 부모 클래스 초기화
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=None,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )
        
        # 데이터셋 검증
        self._validate_dataset()
        
        # PEAR 설정 로깅
        if self.is_world_process_zero():
            self._log_pear_config()
    
    def _validate_pear_params(self):
        """PEAR 파라미터 유효성 검사."""
        valid_modes = ["uniform", "suffix", "block"]
        if self.weighting_mode not in valid_modes:
            raise ValueError(
                f"weighting_mode은 {valid_modes} 중 하나여야 합니다. "
                f"입력값: {self.weighting_mode}"
            )
        
        if self.block_size < 1:
            raise ValueError(f"block_size는 1 이상이어야 합니다. 현재값: {self.block_size}")
        
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma는 (0, 1] 범위여야 합니다. 현재값: {self.gamma}")
    
    def _validate_dataset(self):
        """데이터셋에 behavior_log_probs 컬럼이 있는지 확인."""
        if self.train_dataset is None:
            return
        
        # 첫 샘플 확인
        first_sample = self.train_dataset[0]
        if "behavior_log_probs" not in first_sample:
            raise ValueError(
                "데이터셋에 'behavior_log_probs' 컬럼이 없습니다. "
                "compute_behavior_probs.py를 먼저 실행하여 behavior policy 확률을 계산하세요."
            )
    
    def _log_pear_config(self):
        """PEAR 설정 로깅."""
        print("\n" + "="*70)
        print("PEAR SFT Trainer - Precomputed Behavior Probabilities")
        print("="*70)
        print("✓ Reference model을 메모리에 올리지 않습니다")
        print("✓ 미리 계산된 behavior_log_probs 사용")
        print(f"\nWeighting Mode: {self.weighting_mode}")
        if self.weighting_mode == "block":
            print(f"  └─ Block Size: {self.block_size}")
        print(f"Discount Factor (γ): {self.gamma}")
        print(f"Clip Ratio Range: {self.clip_ratio_range}")
        print(f"Clip Weight Range: {self.clip_weight_range}")
        print("\n핵심 메커니즘:")
        print("  • Target policy (πθ): 학습 중인 모델")
        print("  • Behavior policy (πβ): Precomputed log probabilities")
        print("  • Importance ratio (Δt): exp(log πθ - log πβ)")
        
        if self.weighting_mode == "uniform":
            print("  • Sequence-level weighting")
        elif self.weighting_mode == "suffix":
            print("  • Token-level suffix weighting (논문 권장)")
        elif self.weighting_mode == "block":
            print("  • Block-level weighting")
        
        print("="*70 + "\n")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        PEAR loss를 적용한 compute_loss 메서드.
        
        Precomputed behavior policy probabilities를 사용하여
        importance weighting을 수행합니다.
        """
        # Labels 추출
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("inputs에 'labels' 키가 없습니다.")
        
        # Behavior log probs 추출
        behavior_log_probs = inputs.get("behavior_log_probs")
        if behavior_log_probs is None:
            raise ValueError(
                "inputs에 'behavior_log_probs' 키가 없습니다. "
                "데이터셋에 behavior_log_probs가 포함되어 있는지 확인하세요."
            )
        
        # use_cache를 False로 설정
        inputs["use_cache"] = False
        
        # Forward pass (target policy πθ)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if logits is None:
            raise ValueError("모델 출력에 'logits'가 없습니다.")
        
        # PEAR loss 계산
        loss = self._compute_pear_loss(logits, behavior_log_probs, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_pear_loss(
        self,
        logits: torch.Tensor,
        behavior_log_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        PEAR Weighted Loss 계산 (precomputed version).
        
        Args:
            logits: Target policy logits [batch, seq_len, vocab]
            behavior_log_probs: Precomputed behavior log probs [batch, seq_len]
            labels: Target labels [batch, seq_len]
        
        Returns:
            loss: Weighted loss
        """
        logits = logits.float()
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Labels shift
        labels = F.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = shift_labels.view(-1)
        labels_flat = labels_flat.to(logits.device)
        
        # Behavior log probs도 flatten
        # behavior_log_probs는 이미 shift된 상태 (각 토큰의 log p(yt | x, y<t))
        behavior_log_probs_flat = behavior_log_probs.view(-1)
        behavior_log_probs_flat = behavior_log_probs_flat.to(logits.device)
        
        # 각 배치 샘플별로 처리
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(batch_size):
            start_idx = i * seq_len
            end_idx = (i + 1) * seq_len
            
            sample_logits = logits_flat[start_idx:end_idx]
            sample_labels = labels_flat[start_idx:end_idx]
            sample_behavior_log_probs = behavior_log_probs_flat[start_idx:end_idx]
            
            # Valid token mask
            valid_mask = (sample_labels != -100)
            
            if not valid_mask.any():
                continue
            
            # Per-token loss
            per_token_loss = F.cross_entropy(
                sample_logits, sample_labels, ignore_index=-100, reduction='none'
            )
            
            # Importance weights 계산
            weights = self._compute_importance_weights(
                sample_logits, sample_behavior_log_probs, sample_labels, valid_mask
            )
            
            # Weighted loss
            weighted_loss = (weights * per_token_loss).sum()
            num_valid_tokens = valid_mask.sum()
            
            total_loss += weighted_loss
            total_tokens += num_valid_tokens
        
        # 전체 배치 평균
        if total_tokens > 0:
            loss = total_loss / total_tokens
        else:
            loss = total_loss
        
        return loss
    
    def _compute_importance_weights(
        self,
        logits: torch.Tensor,
        behavior_log_probs: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Importance weights 계산 (precomputed version).
        
        Args:
            logits: [seq_len, vocab] - target policy logits
            behavior_log_probs: [seq_len] - precomputed log πβ(yt|x,y<t)
            labels: [seq_len] - target labels
            valid_mask: [seq_len] - valid token mask
        
        Returns:
            weights: [seq_len] - importance weights
        """
        seq_len = logits.size(0)
        
        # 1. Target policy 확률 계산
        probs = F.softmax(logits, dim=-1)  # πθ
        
        # 2. 정답 토큰의 log 확률 추출
        labels_clamped = labels.clamp(min=0)
        prob_theta = probs.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
        log_prob_theta = torch.log(prob_theta + 1e-10)  # log πθ(yt|...)
        
        # 3. Log-ratio 계산
        # δt = log πθ(yt|...) - log πβ(yt|...)
        log_ratio = log_prob_theta - behavior_log_probs
        
        # 4. Clipping (per-token log-ratio)
        log_ratio = torch.clamp(log_ratio, self.clip_ratio_range[0], self.clip_ratio_range[1])
        
        # 5. Weighting mode에 따라 weight 계산
        if self.weighting_mode == "uniform":
            weights = self._uniform_weighting(log_ratio, valid_mask, seq_len)
        elif self.weighting_mode == "suffix":
            weights = self._suffix_weighting(log_ratio, valid_mask, seq_len)
        elif self.weighting_mode == "block":
            weights = self._block_weighting(log_ratio, valid_mask, seq_len)
        else:
            raise ValueError(f"Unknown weighting_mode: {self.weighting_mode}")
        
        # 6. Invalid token weights는 0으로 설정
        weights = weights * valid_mask.float()
        
        return weights.detach()
    
    def _uniform_weighting(
        self, log_ratio: torch.Tensor, valid_mask: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Sequence-level weighting."""
        total_log_ratio = (log_ratio * valid_mask.float()).sum()
        weight = torch.exp(total_log_ratio)
        weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
        weights = torch.full((seq_len,), weight.item(), device=log_ratio.device)
        return weights
    
    def _suffix_weighting(
        self, log_ratio: torch.Tensor, valid_mask: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Token-level suffix-based weighting."""
        weights = torch.zeros(seq_len, device=log_ratio.device)
        cumsum_log_ratio = 0.0
        
        for t in reversed(range(seq_len)):
            if not valid_mask[t]:
                continue
            
            discount = self.gamma ** (seq_len - t - 1)
            suffix_weight = torch.exp(torch.tensor(cumsum_log_ratio, device=log_ratio.device))
            weight = discount * suffix_weight
            weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
            weights[t] = weight
            cumsum_log_ratio += log_ratio[t].item()
        
        return weights
    
    def _block_weighting(
        self, log_ratio: torch.Tensor, valid_mask: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Block-level weighting."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        weights = torch.zeros(seq_len, device=log_ratio.device)
        cumsum_log_ratio = 0.0
        
        for k in reversed(range(num_blocks)):
            block_start = k * self.block_size
            block_end = min((k + 1) * self.block_size, seq_len)
            block_last_idx = block_end - 1
            
            block_log_ratio = (log_ratio[block_start:block_end] * valid_mask[block_start:block_end].float()).sum()
            discount = self.gamma ** (seq_len - block_last_idx - 1)
            suffix_weight = torch.exp(torch.tensor(cumsum_log_ratio, device=log_ratio.device))
            weight = discount * suffix_weight
            weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
            
            weights[block_start:block_end] = weight
            cumsum_log_ratio += block_log_ratio.item()
        
        return weights


def create_pear_trainer_precomputed(
    model: str | PreTrainedModel,
    train_dataset: Dataset | IterableDataset,
    eval_dataset: Dataset | IterableDataset | None = None,
    output_dir: str = "./pear_output",
    weighting_mode: str = "suffix",
    block_size: int = 1,
    gamma: float = 0.999,
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    **kwargs,
) -> PEARSFTTrainerPrecomputed:
    """
    PEAR Trainer (Precomputed) 헬퍼 함수.
    
    Args:
        model: 학습할 모델
        train_dataset: behavior_log_probs를 포함하는 데이터셋
        eval_dataset: 평가 데이터셋
        output_dir: 출력 디렉토리
        weighting_mode: "uniform", "suffix", "block"
        block_size: Block-level의 블록 크기
        gamma: Discount factor
        learning_rate: 학습률
        num_train_epochs: 에폭 수
        per_device_train_batch_size: 배치 크기
        gradient_accumulation_steps: Gradient accumulation
        **kwargs: 추가 SFTConfig 파라미터
    
    Returns:
        PEARSFTTrainerPrecomputed 인스턴스
    """
    config = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        **kwargs,
    )
    
    trainer = PEARSFTTrainerPrecomputed(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        weighting_mode=weighting_mode,
        block_size=block_size,
        gamma=gamma,
    )
    
    return trainer


if __name__ == "__main__":
    print("PEAR SFT Trainer - Precomputed Behavior Probabilities")
    print("="*70)
    print("Reference model 없이 미리 계산된 확률값으로 PEAR 학습")
    print("="*70)
    print("\n사전 준비:")
    print("1. compute_behavior_probs.py로 behavior_log_probs 계산")
    print("2. 데이터셋에 behavior_log_probs 컬럼 추가")
    print("\n사용 예제:")
    print("""
from for_rl_sft_trainer_precomputed import create_pear_trainer_precomputed
from datasets import load_dataset

# behavior_log_probs가 포함된 데이터셋 로드
dataset = load_dataset("json", data_files="train_with_probs.jsonl", split="train")

# PEAR Trainer 생성
trainer = create_pear_trainer_precomputed(
    model="Qwen/Qwen2.5-1.5B",
    train_dataset=dataset,
    weighting_mode="suffix",
    gamma=0.999,
)

trainer.train()
    """)
    print("\n장점:")
    print("  ✓ Reference model을 메모리에 올리지 않음 (메모리 절약)")
    print("  ✓ 학습 중 reference model forward pass 불필요 (속도 향상)")
    print("  ✓ Behavior policy 확률을 한 번만 계산 (재사용 가능)")
