"""
PEAR (Policy Evaluation-inspired Algorithm for Offline Learning Loss Reweighting) SFT Trainer

논문: Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning
arXiv: 2602.01058v1 [cs.LG] 1 Feb 2026

핵심 아이디어:
- SFT는 단순히 offline 성능만이 아니라 RL 초기화를 잘 준비해야 함
- Behavior policy (데이터 생성)와 Target policy (학습 중인 모델) 간의 distribution mismatch 교정
- Importance sampling으로 offline data를 reweighting하여 online RL과의 정합성 향상

3가지 Variant:
1. Sequence-level: 전체 시퀀스에 동일한 weight
2. Token-level (suffix-based): 각 토큰마다 미래 continuation의 plausibility로 weight (default)
3. Block-level: stability를 위해 block 단위로 weight

수식:
- Importance ratio: Δt = πθ(yt|x,y<t) / πβ(yt|x<t)
- Suffix weight: Gt = γ^(T-t) * ∏(j=t+1 to T) Δj
- Weighted loss: L = Σ sg[Ĝt] * ℓθ(x, y<t, yt)
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


class PEARSFTTrainer(SFTTrainer):
    """
    PEAR 알고리즘을 구현한 SFT Trainer.
    
    Offline SFT 단계에서 importance sampling을 통해 loss를 reweighting하여
    downstream RL을 위한 더 나은 초기화를 제공합니다.
    
    Args:
        model: 학습할 모델 (target policy πθ)
        ref_model: Reference model (behavior policy πβ).
            None일 경우 학습 시작 시점의 model을 복사하여 사용합니다.
        weighting_mode: Importance weighting 방식
            - "uniform": Sequence-level weighting (전체 시퀀스 동일 weight)
            - "suffix": Token-level suffix-based weighting (default, 논문 권장)
            - "block": Block-level weighting (stability 향상)
        block_size: Block-level mode에서 블록 크기 (default: 1 = token-level)
        gamma: Discount factor for future tokens (default: 0.999)
        clip_ratio_range: Per-token log-ratio clipping range [lower, upper] (default: [-0.08, 0.3])
        clip_weight_range: Final weight clipping range [min, max] (default: [0.1, 10.0])
        use_negative_data: Negative examples을 사용할지 여부 (default: False)
        negative_weight: Negative examples의 weight (default: 1.0)
    
    Examples:
        >>> from for_rl_sft_trainer import PEARSFTTrainer
        >>> from trl import SFTConfig
        >>> 
        >>> # PEAR 기본 설정 (Token-level suffix weighting)
        >>> config = SFTConfig(
        ...     output_dir="./pear_output",
        ...     learning_rate=1e-5,
        ...     num_train_epochs=1,
        ... )
        >>> 
        >>> trainer = PEARSFTTrainer(
        ...     model="Qwen/Qwen2.5-0.5B-Instruct",
        ...     args=config,
        ...     train_dataset=dataset,
        ...     weighting_mode="suffix",  # Token-level suffix weighting
        ...     gamma=0.999,
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
        ref_model: Optional[PreTrainedModel] = None,
        weighting_mode: str = "suffix",
        block_size: int = 1,
        gamma: float = 0.999,
        clip_ratio_range: tuple[float, float] = (-0.08, 0.3),
        clip_weight_range: tuple[float, float] = (0.1, 10.0),
        use_negative_data: bool = False,
        negative_weight: float = 1.0,
    ):
        """PEAR SFT Trainer 초기화."""
        
        # PEAR 파라미터 저장
        self.ref_model = ref_model
        self.weighting_mode = weighting_mode
        self.block_size = block_size
        self.gamma = gamma
        self.clip_ratio_range = clip_ratio_range
        self.clip_weight_range = clip_weight_range
        self.use_negative_data = use_negative_data
        self.negative_weight = negative_weight
        
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
            compute_loss_func=None,  # compute_loss 메서드를 직접 오버라이드
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )
        
        # Reference model 초기화 (behavior policy πβ)
        self._setup_ref_model()
        
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
        
        if len(self.clip_ratio_range) != 2:
            raise ValueError("clip_ratio_range는 [lower, upper] 형태여야 합니다.")
        
        if len(self.clip_weight_range) != 2:
            raise ValueError("clip_weight_range는 [min, max] 형태여야 합니다.")
    
    def _setup_ref_model(self):
        """Reference model (behavior policy) 설정."""
        if self.ref_model is None:
            # Reference model이 없으면 현재 model을 복사
            if self.is_world_process_zero():
                print("⚠️  Reference model이 제공되지 않아 현재 모델을 복사합니다.")
                print("    실제 사용 시에는 데이터 생성에 사용한 모델을 ref_model로 전달하세요.")
            
            # 모델 복사 (gradient 계산 불필요)
            self.ref_model = type(self.model)(self.model.config)
            self.ref_model.load_state_dict(self.model.state_dict())
        
        # Reference model을 evaluation mode로 설정 및 gradient 비활성화
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 같은 device로 이동
        if hasattr(self.model, 'device'):
            self.ref_model = self.ref_model.to(self.model.device)
    
    def _log_pear_config(self):
        """PEAR 설정 로깅."""
        print("\n" + "="*70)
        print("PEAR SFT Trainer - RL을 위한 Offline 학습")
        print("="*70)
        print(f"Weighting Mode: {self.weighting_mode}")
        if self.weighting_mode == "block":
            print(f"  └─ Block Size: {self.block_size}")
        print(f"Discount Factor (γ): {self.gamma}")
        print(f"Clip Ratio Range: {self.clip_ratio_range}")
        print(f"Clip Weight Range: {self.clip_weight_range}")
        print(f"Use Negative Data: {self.use_negative_data}")
        if self.use_negative_data:
            print(f"  └─ Negative Weight: {self.negative_weight}")
        print("\n핵심 메커니즘:")
        print("  • Target policy (πθ): 학습 중인 모델")
        print("  • Behavior policy (πβ): Reference model")
        print("  • Importance ratio (Δt): πθ(yt|x,y<t) / πβ(yt|x,y<t)")
        
        if self.weighting_mode == "uniform":
            print("  • Sequence-level weighting: 전체 시퀀스에 동일한 weight")
        elif self.weighting_mode == "suffix":
            print("  • Token-level suffix weighting: 미래 continuation 고려 (논문 권장)")
        elif self.weighting_mode == "block":
            print("  • Block-level weighting: Stability 향상")
        
        print("="*70 + "\n")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        PEAR loss를 적용한 compute_loss 메서드.
        
        Importance sampling을 통해 offline data를 reweighting하여
        online RL과의 distribution mismatch를 교정합니다.
        """
        # Labels 추출
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("inputs에 'labels' 키가 없습니다.")
        
        # use_cache를 False로 설정
        inputs["use_cache"] = False
        
        # Forward pass (target policy πθ)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if logits is None:
            raise ValueError("모델 출력에 'logits'가 없습니다.")
        
        # Reference model forward pass (behavior policy πβ)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.get("logits")
            
            if ref_logits is None:
                raise ValueError("Reference 모델 출력에 'logits'가 없습니다.")
        
        # PEAR loss 계산
        loss = self._compute_pear_loss(logits, ref_logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_pear_loss(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        PEAR Weighted Loss 계산.
        
        Args:
            logits: Target policy logits [batch, seq_len, vocab]
            ref_logits: Behavior policy logits [batch, seq_len, vocab]
            labels: Target labels [batch, seq_len]
        
        Returns:
            loss: Weighted loss
        """
        logits = logits.float()
        ref_logits = ref_logits.float()
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Labels shift (다음 토큰 예측)
        labels = F.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for computation
        logits_flat = logits.view(-1, vocab_size)
        ref_logits_flat = ref_logits.view(-1, vocab_size)
        labels_flat = shift_labels.view(-1)
        labels_flat = labels_flat.to(logits.device)
        
        # 각 배치 샘플별로 처리 (시퀀스 단위로 importance weight 계산 필요)
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(batch_size):
            start_idx = i * seq_len
            end_idx = (i + 1) * seq_len
            
            sample_logits = logits_flat[start_idx:end_idx]  # [seq_len, vocab]
            sample_ref_logits = ref_logits_flat[start_idx:end_idx]  # [seq_len, vocab]
            sample_labels = labels_flat[start_idx:end_idx]  # [seq_len]
            
            # Valid token mask (ignore_index가 아닌 토큰들)
            valid_mask = (sample_labels != -100)
            
            if not valid_mask.any():
                continue
            
            # Per-token loss 계산 (reduction='none')
            per_token_loss = F.cross_entropy(
                sample_logits, sample_labels, ignore_index=-100, reduction='none'
            )  # [seq_len]
            
            # Importance weights 계산
            weights = self._compute_importance_weights(
                sample_logits, sample_ref_logits, sample_labels, valid_mask
            )  # [seq_len]
            
            # Weighted loss
            weighted_loss = (weights * per_token_loss).sum()
            num_valid_tokens = valid_mask.sum()
            
            total_loss += weighted_loss
            total_tokens += num_valid_tokens
        
        # 전체 배치에 대한 평균
        if total_tokens > 0:
            loss = total_loss / total_tokens
        else:
            loss = total_loss
        
        return loss
    
    def _compute_importance_weights(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Importance weights 계산 (Algorithm 1 구현).
        
        Args:
            logits: [seq_len, vocab] - target policy logits
            ref_logits: [seq_len, vocab] - behavior policy logits
            labels: [seq_len] - target labels
            valid_mask: [seq_len] - valid token mask
        
        Returns:
            weights: [seq_len] - importance weights
        """
        seq_len = logits.size(0)
        
        # 1. Probabilities 계산
        probs = F.softmax(logits, dim=-1)  # πθ
        ref_probs = F.softmax(ref_logits, dim=-1)  # πβ
        
        # 2. 정답 토큰의 확률 추출
        labels_clamped = labels.clamp(min=0)
        
        prob_theta = probs.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)  # πθ(yt|...)
        prob_beta = ref_probs.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)  # πβ(yt|...)
        
        # 3. Log-ratio 계산 (numerical stability)
        # δt = log(πθ(yt|...)) - log(πβ(yt|...))
        log_ratio = torch.log(prob_theta + 1e-10) - torch.log(prob_beta + 1e-10)
        
        # 4. Clipping (per-token log-ratio)
        log_ratio = torch.clamp(log_ratio, self.clip_ratio_range[0], self.clip_ratio_range[1])
        
        # 5. Weighting mode에 따라 weight 계산
        if self.weighting_mode == "uniform":
            # Sequence-level: 전체 시퀀스의 importance ratio
            weights = self._uniform_weighting(log_ratio, valid_mask, seq_len)
        
        elif self.weighting_mode == "suffix":
            # Token-level suffix-based: 각 토큰의 미래 continuation 고려
            weights = self._suffix_weighting(log_ratio, valid_mask, seq_len)
        
        elif self.weighting_mode == "block":
            # Block-level: stability를 위해 block 단위로 처리
            weights = self._block_weighting(log_ratio, valid_mask, seq_len)
        
        else:
            raise ValueError(f"Unknown weighting_mode: {self.weighting_mode}")
        
        # 6. Invalid token weights는 0으로 설정 (loss에 영향 없음)
        weights = weights * valid_mask.float()
        
        return weights.detach()  # stop gradient
    
    def _uniform_weighting(
        self, log_ratio: torch.Tensor, valid_mask: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        Sequence-level weighting (논문 §3.3).
        
        전체 시퀀스에 대한 importance ratio를 모든 토큰에 동일하게 적용.
        w = ∏(t=1 to T) Δt = exp(Σ δt)
        """
        # 전체 시퀀스 log-ratio 합
        total_log_ratio = (log_ratio * valid_mask.float()).sum()
        
        # Sequence-level weight
        weight = torch.exp(total_log_ratio)
        
        # Clipping
        weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
        
        # 모든 토큰에 동일한 weight
        weights = torch.full((seq_len,), weight.item(), device=log_ratio.device)
        
        return weights
    
    def _suffix_weighting(
        self, log_ratio: torch.Tensor, valid_mask: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        Token-level suffix-based weighting (논문 §3.4, default).
        
        각 토큰 t에 대해 미래 continuation (t+1부터 T까지)의 importance ratio를 weight로 사용.
        Gt = γ^(T-t) * ∏(j=t+1 to T) Δj
        """
        weights = torch.zeros(seq_len, device=log_ratio.device)
        
        # Backward scan (Algorithm 1, line 14-20)
        cumsum_log_ratio = 0.0  # Σ(j=k+1 to T) δj
        
        for t in reversed(range(seq_len)):
            if not valid_mask[t]:
                continue
            
            # Discount factor: γ^(T-t)
            discount = self.gamma ** (seq_len - t - 1)
            
            # Suffix importance ratio: exp(Σ(j=t+1 to T) δj)
            suffix_weight = torch.exp(torch.tensor(cumsum_log_ratio, device=log_ratio.device))
            
            # Combined weight with discount
            weight = discount * suffix_weight
            
            # Clipping
            weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
            
            weights[t] = weight
            
            # Update cumsum for next iteration
            cumsum_log_ratio += log_ratio[t].item()
        
        return weights
    
    def _block_weighting(
        self, log_ratio: torch.Tensor, valid_mask: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        Block-level weighting (논문 §3.5).
        
        Stability를 위해 토큰들을 블록으로 나누고, 블록 내 모든 토큰에 동일한 weight 적용.
        Block size B를 사용하여 variance 감소.
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        weights = torch.zeros(seq_len, device=log_ratio.device)
        
        # Backward scan by blocks
        cumsum_log_ratio = 0.0
        
        for k in reversed(range(num_blocks)):
            block_start = k * self.block_size
            block_end = min((k + 1) * self.block_size, seq_len)
            
            # Block의 마지막 토큰 인덱스
            block_last_idx = block_end - 1
            
            # Block 내 log-ratio 합계
            block_log_ratio = (log_ratio[block_start:block_end] * valid_mask[block_start:block_end].float()).sum()
            
            # Discount factor: γ^(T - block_last_idx)
            discount = self.gamma ** (seq_len - block_last_idx - 1)
            
            # Suffix importance ratio
            suffix_weight = torch.exp(torch.tensor(cumsum_log_ratio, device=log_ratio.device))
            
            # Block weight
            weight = discount * suffix_weight
            weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
            
            # 블록 내 모든 토큰에 동일한 weight
            weights[block_start:block_end] = weight
            
            # Update cumsum
            cumsum_log_ratio += block_log_ratio.item()
        
        return weights


def create_pear_trainer(
    model: str | PreTrainedModel,
    train_dataset: Dataset | IterableDataset,
    eval_dataset: Dataset | IterableDataset | None = None,
    ref_model: Optional[PreTrainedModel] = None,
    output_dir: str = "./pear_output",
    weighting_mode: str = "suffix",
    block_size: int = 1,
    gamma: float = 0.999,
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    **kwargs,
) -> PEARSFTTrainer:
    """
    PEAR Trainer를 쉽게 생성하는 헬퍼 함수.
    
    Args:
        model: 학습할 모델 (target policy)
        train_dataset: 학습 데이터셋
        eval_dataset: 평가 데이터셋 (optional)
        ref_model: Reference model (behavior policy, optional)
        output_dir: 출력 디렉토리
        weighting_mode: "uniform", "suffix", "block" 중 선택
        block_size: Block-level mode의 블록 크기
        gamma: Discount factor
        learning_rate: 학습률 (논문: 1e-5 for math, 3e-5 for logic games)
        num_train_epochs: 에폭 수 (논문: 1 epoch)
        per_device_train_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: gradient accumulation 스텝
        **kwargs: 추가 SFTConfig 파라미터
    
    Returns:
        PEARSFTTrainer 인스턴스
    
    Examples:
        >>> from for_rl_sft_trainer import create_pear_trainer
        >>> from datasets import load_dataset
        >>> 
        >>> dataset = load_dataset("your/dataset")
        >>> 
        >>> # Token-level suffix weighting (논문 권장)
        >>> trainer = create_pear_trainer(
        ...     model="Qwen/Qwen2.5-0.5B-Instruct",
        ...     train_dataset=dataset["train"],
        ...     weighting_mode="suffix",
        ...     gamma=0.999,
        ... )
        >>> trainer.train()
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
    
    trainer = PEARSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ref_model=ref_model,
        weighting_mode=weighting_mode,
        block_size=block_size,
        gamma=gamma,
    )
    
    return trainer


if __name__ == "__main__":
    # 사용 예제
    print("PEAR SFT Trainer - RL을 위한 Offline 학습")
    print("="*70)
    print("논문: Good SFT Optimizes for SFT, Better SFT Prepares for RL")
    print("="*70)
    print("\n사용 예제:")
    print("""
from for_rl_sft_trainer import PEARSFTTrainer, create_pear_trainer
from datasets import load_dataset

# 방법 1: 직접 생성
trainer = PEARSFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
    weighting_mode="suffix",  # Token-level suffix weighting (default)
    gamma=0.999,
)

# 방법 2: 헬퍼 함수 사용
trainer = create_pear_trainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
    ref_model=behavior_policy_model,  # 데이터 생성에 사용한 모델
    weighting_mode="suffix",
)

trainer.train()

# 학습 후 RL 초기화로 사용
# 이 checkpoint는 offline 성능보다 online RL 성능을 더 잘 향상시킵니다!
    """)
    print("\n주요 파라미터:")
    print("  • weighting_mode:")
    print("    - 'uniform': Sequence-level (전체 시퀀스 동일 weight)")
    print("    - 'suffix': Token-level suffix (미래 continuation 고려, 권장)")
    print("    - 'block': Block-level (stability 향상)")
    print("  • gamma: Discount factor (default: 0.999)")
    print("  • block_size: Block mode의 블록 크기 (default: 1)")
    print("  • ref_model: Behavior policy model (데이터 생성 모델)")
