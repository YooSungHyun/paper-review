"""
ProFit SFT Trainer - TRL 기반 구현

논문: ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection
GitHub: https://github.com/Utaotao/ProFit

핵심 아이디어:
- 높은 확률 토큰 = 핵심 논리/의미
- 낮은 확률 토큰 = 대체 가능한 표현
- 낮은 확률 토큰을 선택적으로 마스킹하여 표면적 과적합 방지
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


class ProFitSFTTrainer(SFTTrainer):
    """
    ProFit 알고리즘을 구현한 SFT Trainer.
    
    TRL의 SFTTrainer를 상속받아 compute_loss 메서드를 오버라이드하여
    확률 기반 토큰 선택을 통한 selective masking을 구현합니다.
    
    Args:
        prob_threshold (float or list[float]):
            토큰 마스킹을 위한 확률 임계값.
            - threshold_direction="higher"일 때: 이 값보다 낮은 확률의 토큰 마스킹
            - threshold_direction="lower"일 때: 이 값보다 높은 확률의 토큰 마스킹
            - threshold_direction="middle"일 때: [lower, upper] 범위 밖의 토큰 마스킹
        threshold_direction (str):
            마스킹 방향. "higher", "lower", "middle", "random" 중 선택.
            - "higher": 낮은 확률 토큰 마스킹 (논문의 기본 설정)
            - "lower": 높은 확률 토큰 마스킹
            - "middle": 중간 범위 토큰만 학습
            - "random": 랜덤 마스킹 (baseline)
        use_profit_loss (bool):
            ProFit loss 사용 여부. False일 경우 일반 SFT와 동일.
    
    Examples:
        >>> from profit_sft_trainer import ProFitSFTTrainer
        >>> from trl import SFTConfig
        >>> 
        >>> # ProFit 설정: 확률 0.3 미만의 토큰 마스킹
        >>> config = SFTConfig(
        ...     output_dir="./output",
        ...     learning_rate=2e-5,
        ...     num_train_epochs=3,
        ... )
        >>> 
        >>> trainer = ProFitSFTTrainer(
        ...     model="Qwen/Qwen2.5-0.5B-Instruct",
        ...     args=config,
        ...     train_dataset=dataset,
        ...     prob_threshold=0.3,
        ...     threshold_direction="higher",
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
        # ProFit 전용 파라미터
        prob_threshold: Union[float, list[float]] = 0.3,
        threshold_direction: str = "higher",
        use_profit_loss: bool = True,
    ):
        """ProFit SFT Trainer 초기화."""
        
        # ProFit 파라미터 저장
        self.prob_threshold = prob_threshold if isinstance(prob_threshold, list) else [prob_threshold]
        self.threshold_direction = threshold_direction
        self.use_profit_loss = use_profit_loss
        
        # 파라미터 검증
        self._validate_profit_params()
        
        # 부모 클래스 초기화 (compute_loss_func는 None으로 설정)
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
        
        # ProFit 설정 로깅
        if self.use_profit_loss and self.is_world_process_zero():
            print("\n" + "="*60)
            print("ProFit SFT Trainer 초기화 완료")
            print("="*60)
            print(f"Probability Threshold: {self.prob_threshold}")
            print(f"Threshold Direction: {self.threshold_direction}")
            print(f"마스킹 전략: ", end="")
            if self.threshold_direction == "higher":
                print(f"확률 < {self.prob_threshold[0]}인 토큰 마스킹 (낮은 확률 토큰 제거)")
            elif self.threshold_direction == "lower":
                print(f"확률 > {self.prob_threshold[0]}인 토큰 마스킹 (높은 확률 토큰 제거)")
            elif self.threshold_direction == "middle":
                print(f"확률 < {self.prob_threshold[0]} 또는 > {self.prob_threshold[1]}인 토큰 마스킹")
            elif self.threshold_direction == "random":
                print(f"{self.prob_threshold[0]*100:.1f}% 확률로 랜덤 마스킹")
            print("="*60 + "\n")
    
    def _validate_profit_params(self):
        """ProFit 파라미터 유효성 검사."""
        valid_directions = ["higher", "lower", "middle", "random"]
        if self.threshold_direction not in valid_directions:
            raise ValueError(
                f"threshold_direction은 {valid_directions} 중 하나여야 합니다. "
                f"입력값: {self.threshold_direction}"
            )
        
        if self.threshold_direction == "middle":
            if len(self.prob_threshold) != 2:
                raise ValueError(
                    "threshold_direction='middle'일 때 prob_threshold는 [lower, upper] 형태의 리스트여야 합니다."
                )
            if self.prob_threshold[0] >= self.prob_threshold[1]:
                raise ValueError(
                    f"middle 모드에서 lower < upper 조건을 만족해야 합니다. "
                    f"현재값: {self.prob_threshold}"
                )
        
        for threshold in self.prob_threshold:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"prob_threshold는 0.0과 1.0 사이의 값이어야 합니다. 현재값: {threshold}"
                )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        ProFit loss를 적용한 compute_loss 메서드.
        
        일반 SFT와 동일하게 forward pass를 수행하지만,
        loss 계산 시 토큰의 확률에 따라 선택적으로 마스킹합니다.
        """
        if not self.use_profit_loss:
            # ProFit을 사용하지 않을 경우 부모 클래스의 메서드 호출
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        # Labels 추출 (shift_labels가 있으면 우선 사용)
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("inputs에 'labels' 키가 없습니다.")
        
        # use_cache를 False로 설정 (gradient checkpointing과 호환)
        inputs["use_cache"] = False
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if logits is None:
            # logits가 없으면 모델이 이미 loss를 계산한 경우
            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("모델 출력에 'logits'와 'loss'가 모두 없습니다.")
            return (loss, outputs) if return_outputs else loss
        
        # ProFit loss 계산
        loss = self._compute_profit_loss(logits, labels, num_items_in_batch)
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_profit_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        ProFit Cross Entropy Loss 계산.
        
        Args:
            logits: 모델 출력 logits [batch_size, seq_len, vocab_size]
            labels: 타겟 labels [batch_size, seq_len]
            num_items_in_batch: 전체 배치 크기 (gradient accumulation 고려)
        
        Returns:
            loss: 계산된 loss 값
        """
        logits = logits.float()
        vocab_size = logits.size(-1)
        
        # Labels shift (다음 토큰 예측)
        # [batch, seq_len] -> [batch, seq_len+1] (padding) -> [batch, seq_len] (shift)
        labels = F.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)
        
        # ProFit Cross Entropy
        loss = self._profit_cross_entropy(
            logits,
            shift_labels,
            num_items_in_batch,
            self.prob_threshold,
            self.threshold_direction,
        )
        
        return loss
    
    def _profit_cross_entropy(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
        prob_threshold: list[float] = [0.3],
        threshold_direction: str = "higher",
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        확률 기반 선택적 마스킹을 적용한 Cross Entropy Loss.
        
        Args:
            source: logits [N, vocab_size]
            target: labels [N]
            num_items_in_batch: 전체 배치 아이템 수 (사용 안 함, 호환성 유지용)
            prob_threshold: 확률 임계값
            threshold_direction: 마스킹 방향
            ignore_index: 무시할 인덱스 값
        
        Returns:
            loss: 계산된 loss 값
        """
        # 1. 확률 계산
        probs = F.softmax(source, dim=-1)
        
        # 2. 정답 토큰의 확률 추출
        target_for_gather = target.clone().clamp(min=0)  # -100을 0으로 변환 (gather용)
        prob_of_correct_token = probs.gather(
            dim=-1, index=target_for_gather.unsqueeze(-1)
        ).squeeze(-1)
        
        # 3. 새로운 타겟 생성 (마스킹 적용)
        new_target = target.clone()
        
        # 4. Threshold에 따라 마스킹 조건 설정
        if threshold_direction == "higher":
            # 낮은 확률 토큰 마스킹 (ProFit 기본 설정)
            mask_condition = (prob_of_correct_token.detach() < prob_threshold[0])
        elif threshold_direction == "lower":
            # 높은 확률 토큰 마스킹
            mask_condition = (prob_of_correct_token.detach() > prob_threshold[0])
        elif threshold_direction == "middle":
            # 중간 범위 밖 토큰 마스킹
            mask_condition = (
                (prob_of_correct_token.detach() < prob_threshold[0]) |
                (prob_of_correct_token.detach() > prob_threshold[1])
            )
        elif threshold_direction == "random":
            # 랜덤 마스킹 (baseline)
            mask_condition = torch.rand_like(prob_of_correct_token) < prob_threshold[0]
        else:
            raise ValueError(f"알 수 없는 threshold_direction: {threshold_direction}")
        
        # 5. 마스킹 적용
        new_target[mask_condition] = ignore_index
        
        # 6. 유효한 토큰 개수 계산 (원래 마스킹 + ProFit 마스킹 모두 고려)
        # transformers의 LabelSmoother와 동일한 방식
        num_active_elements = (new_target != ignore_index).sum()
        
        # 7. Loss 계산 (reduction="sum"으로 먼저 합산)
        loss = F.cross_entropy(
            source, new_target, ignore_index=ignore_index, reduction="sum"
        )
        
        # 8. 토큰 단위 평균 계산 (유효한 토큰 개수로 나누기)
        loss = loss / torch.clamp(num_active_elements, min=1)
        
        return loss


def create_profit_trainer(
    model: str | PreTrainedModel,
    train_dataset: Dataset | IterableDataset,
    eval_dataset: Dataset | IterableDataset | None = None,
    output_dir: str = "./profit_output",
    prob_threshold: Union[float, list[float]] = 0.3,
    threshold_direction: str = "higher",
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    **kwargs,
) -> ProFitSFTTrainer:
    """
    ProFit Trainer를 쉽게 생성하는 헬퍼 함수.
    
    Args:
        model: 모델 이름 또는 PreTrainedModel
        train_dataset: 학습 데이터셋
        eval_dataset: 평가 데이터셋 (optional)
        output_dir: 출력 디렉토리
        prob_threshold: ProFit 확률 임계값
        threshold_direction: 마스킹 방향
        learning_rate: 학습률
        num_train_epochs: 에폭 수
        per_device_train_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: gradient accumulation 스텝
        **kwargs: 추가 SFTConfig 파라미터
    
    Returns:
        ProFitSFTTrainer 인스턴스
    
    Examples:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")
        >>> 
        >>> trainer = create_profit_trainer(
        ...     model="Qwen/Qwen2.5-0.5B-Instruct",
        ...     train_dataset=dataset,
        ...     prob_threshold=0.3,
        ...     threshold_direction="higher",
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
    
    trainer = ProFitSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prob_threshold=prob_threshold,
        threshold_direction=threshold_direction,
    )
    
    return trainer


if __name__ == "__main__":
    # 간단한 사용 예제
    print("ProFit SFT Trainer 모듈")
    print("="*60)
    print("사용 예제:")
    print("""
from profit_sft_trainer import ProFitSFTTrainer, create_profit_trainer
from datasets import load_dataset

# 방법 1: 직접 생성
trainer = ProFitSFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
    prob_threshold=0.3,
    threshold_direction="higher",
)

# 방법 2: 헬퍼 함수 사용
trainer = create_profit_trainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
    prob_threshold=0.3,
)

trainer.train()
    """)
