"""
ProFit SFT Trainer (patched) - TRL 기반 구현

논문: ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection

이 버전의 목표:
1) ProFit의 확률 기반 hard masking 게이트는 stop-gradient(detach)로 계산
2) 마스킹 이후 active token만 학습
3) 분산 학습에서 loss_sum / active_count를 all-reduce(sum)하여
   "전역 token-weighted 평균 loss"를 반환 (길이/마스킹 비율/packing 차이에도 안정)

(4) 실전 개선점 반영:
- compute_loss에서 inputs dict in-place 수정 방지
- prob_threshold에 가변 기본값(list) 미사용 (float 기본값 유지)
- 마스킹 비율/active token 수를 주기적으로 로깅(기본 100 step)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _all_reduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    분산 환경이면 x를 all-reduce(sum)해서 반환.

    핵심 설계:
    1. Forward: all_reduce로 global sum 계산 (detached 사용)
    2. Backward: 원본 x를 통해 gradient 흐름 유지
    3. DDP/DeepSpeed가 자동으로 gradient all_reduce 수행

    트릭: detached_global_sum + (x - x.detach())
    - Forward 값: detached_global_sum + 0 = global_sum
    - Backward: (x - x.detach())를 통해 gradient는 x로만 흐름
    """
    if _dist_is_initialized():
        # 1. 값만 all_reduce (gradient 끊음)
        x_global = x.detach().clone()
        torch.distributed.all_reduce(x_global, op=torch.distributed.ReduceOp.SUM)

        # 2. Gradient graph 유지
        # Forward: x_global + (x - x.detach()) = x_global + 0 = x_global
        # Backward: gradient는 (x - x.detach())의 x를 통해서만 흐름
        return x_global + (x - x.detach())
    return x


class ProFitSFTTrainer(SFTTrainer):
    """
    ProFit 알고리즘을 구현한 SFT Trainer.

    prob_threshold (float or list[float]):
        - "higher":  p(correct) < tau 인 토큰을 마스킹 (논문 기본)
        - "lower":   p(correct) > tau 인 토큰을 마스킹
        - "middle":  [low, high] 범위 밖 토큰 마스킹
        - "random":  tau 확률로 랜덤 마스킹 (baseline)

    threshold_direction: {"higher","lower","middle","random"}
    use_profit_loss: ProFit 적용 여부
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
        # ProFit params
        prob_threshold: Union[float, list[float]] = 0.3,  # ✅ list 기본값(가변) 사용하지 않음
        threshold_direction: str = "higher",
        use_profit_loss: bool = True,
        # logging
        profit_log_every: int = 100,  # ✅ active ratio 로깅 주기(steps)
        # 🆕 템플릿 토큰 강제 학습
        force_include_tokens: Optional[list[str]] = None,  # 템플릿 토큰 텍스트 리스트 (일반 텍스트)
        force_include_patterns: Optional[list[str]] = None,  # 🆕 정규식 패턴 리스트
        use_pattern_masking: bool = False,  # 🆕 패턴을 정규식으로 처리할지 여부
    ):
        self.prob_threshold = prob_threshold if isinstance(prob_threshold, list) else [prob_threshold]
        self.threshold_direction = threshold_direction
        self.use_profit_loss = use_profit_loss
        self.profit_log_every = int(profit_log_every)
        self.force_include_tokens = force_include_tokens or []
        self.force_include_patterns = force_include_patterns or []
        self.use_pattern_masking = use_pattern_masking

        # 마지막 스텝 통계 저장용
        self._last_profit_stats: dict[str, float] = {}

        # 강제 포함 토큰 ID 저장 (tokenizer 초기화 후 설정)
        self._force_include_token_ids: set[int] = set()

        self._validate_profit_params()

        # 🆕 정규식 패턴 모드: super().__init__() 전에 데이터셋에 force_include_mask 추가
        # processing_class를 tokenizer로 직접 사용
        if (
            self.use_pattern_masking
            and self.force_include_patterns
            and train_dataset is not None
            and processing_class is not None
        ):
            from accelerate import PartialState

            # main process에서만 출력
            if PartialState().is_main_process:
                print("\n" + "=" * 60)
                print("정규식 패턴 기반 마스킹 적용 중...")
                print("=" * 60)

            train_dataset = self._add_pattern_masks_to_dataset(train_dataset, processing_class)
            
            # 🆕 Custom collator 자동 설정 (data_collator가 None일 때만)
            if data_collator is None:
                # padding_free 설정 확인
                # TRL은 padding_free=True + custom collator 조합을 막아놨으므로
                # args.padding_free를 False로 변경하고 collator에서 직접 처리
                is_padding_free = getattr(args, "padding_free", False)
                
                if is_padding_free:
                    from profit_data_collator import ProFitDataCollatorPaddingFree
                    data_collator = ProFitDataCollatorPaddingFree(
                        tokenizer=processing_class,
                    )
                    # TRL 제약 우회: padding_free=False로 변경
                    args.padding_free = False
                    if PartialState().is_main_process:
                        print("✓ ProFitDataCollatorPaddingFree 설정됨")
                        print("  (args.padding_free=False로 변경 - TRL 제약 우회)")
                else:
                    from profit_data_collator import ProFitDataCollatorForLanguageModeling
                    data_collator = ProFitDataCollatorForLanguageModeling(
                        tokenizer=processing_class,
                        mlm=False,
                    )
                    if PartialState().is_main_process:
                        print("✓ ProFitDataCollatorForLanguageModeling 설정됨 (padding 환경)")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=None,  # compute_loss override
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        # HF Trainer가 num_items_in_batch 등을 전달해도, 이 구현은 쓰지 않음.
        self.model_accepts_loss_kwargs = True

        # 🆕 tokenizer를 사용하여 강제 포함 토큰을 ID로 변환 (일반 텍스트 모드)
        if self.force_include_tokens and self.tokenizer is not None and not self.use_pattern_masking:
            for token_text in self.force_include_tokens:
                # 각 토큰 텍스트를 인코딩하여 ID 추출
                token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
                self._force_include_token_ids.update(token_ids)

        if self.use_profit_loss and self.is_world_process_zero():
            print("\n" + "=" * 60)
            print("ProFitSFTTrainer (patched) initialized")
            print("=" * 60)
            print(f"prob_threshold={self.prob_threshold}")
            print(f"threshold_direction={self.threshold_direction}")
            print("loss = global_sum(nll_active_tokens) / global_sum(#active_tokens)")
            print("stop-grad gate = detach(p_correct)")
            print(f"profit_log_every={self.profit_log_every} steps")

            if self.use_pattern_masking and self.force_include_patterns:
                print(f"\n[정규식 패턴 모드]")
                print(f"force_include_patterns={self.force_include_patterns}")
                print("  → 정규식 패턴 매칭으로 토큰 강제 포함")
            elif self.force_include_tokens:
                print(f"\n[일반 텍스트 모드]")
                print(f"force_include_tokens={self.force_include_tokens}")
                print(f"force_include_token_ids={sorted(self._force_include_token_ids)}")

            print("=" * 60 + "\n")

    def _add_pattern_masks_to_dataset(
        self,
        dataset: Dataset | IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Dataset | IterableDataset:
        """
        정규식 패턴을 사용하여 데이터셋에 force_include_mask를 추가합니다.

        Args:
            dataset: 원본 데이터셋
            tokenizer: 토크나이저

        Returns:
            force_include_mask가 추가된 데이터셋
        """
        from utils.masking_utils import create_force_include_mask
        from accelerate import PartialState

        def add_mask(example):
            """각 샘플에 force_include_mask 추가"""
            # input_ids를 디코딩하여 원본 텍스트 복원
            input_ids = example["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)

            # 디코딩 (skip_special_tokens=False로 특수 토큰도 포함)
            text = tokenizer.decode(input_ids, skip_special_tokens=False)

            # 정규식 패턴 매칭으로 마스크 생성
            force_mask = create_force_include_mask(
                text=text,
                tokenizer=tokenizer,
                patterns=self.force_include_patterns,
                add_special_tokens=False,
            )

            # 길이가 맞지 않으면 조정
            if len(force_mask) < len(input_ids):
                force_mask += [False] * (len(input_ids) - len(force_mask))
            elif len(force_mask) > len(input_ids):
                force_mask = force_mask[: len(input_ids)]

            example["force_include_mask"] = force_mask
            return example

        # 데이터셋에 마스크 추가
        if isinstance(dataset, Dataset):
            dataset = dataset.map(add_mask, desc="Adding pattern masks")
        else:
            # IterableDataset은 map 지원 안 함 - 경고만 출력
            if PartialState().is_main_process:
                print("⚠️  경고: IterableDataset은 정규식 패턴 마스킹을 미리 적용할 수 없습니다.")
                print("    → 학습 중 동적으로 처리되지만 성능이 저하될 수 있습니다.")

        return dataset

    def _validate_profit_params(self) -> None:
        valid = {"higher", "lower", "middle", "random"}
        if self.threshold_direction not in valid:
            raise ValueError(f"threshold_direction must be one of {sorted(valid)}; got {self.threshold_direction}")

        if self.threshold_direction == "middle":
            if len(self.prob_threshold) != 2:
                raise ValueError("threshold_direction='middle' requires prob_threshold=[low, high]")
            if self.prob_threshold[0] >= self.prob_threshold[1]:
                raise ValueError(f"middle mode requires low < high; got {self.prob_threshold}")

        for t in self.prob_threshold:
            if not (0.0 <= t <= 1.0):
                raise ValueError(f"prob_threshold must be in [0,1]; got {t}")

        if self.profit_log_every < 0:
            raise ValueError(f"profit_log_every must be >= 0; got {self.profit_log_every}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        ProFit loss를 적용한 compute_loss.
        - forward로 logits 획득
        - (shift) next-token prediction CE
        - ProFit 마스킹 적용
        - 전역 token-weighted 평균 loss 반환
        """
        if not self.use_profit_loss:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # ✅ inputs를 in-place 수정하지 않기
        inputs = dict(inputs)

        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("inputs must contain 'labels'")

        # gradient checkpointing 호환
        inputs["use_cache"] = False

        outputs = model(**inputs)
        logits = outputs.get("logits", None)
        if logits is None:
            loss = outputs.get("loss", None)
            if loss is None:
                raise ValueError("model outputs contain neither 'logits' nor 'loss'")
            return (loss, outputs) if return_outputs else loss

        # 🆕 데이터셋에서 force_include_mask 가져오기 (정규식 패턴으로 생성된 경우)
        force_include_mask = inputs.get("force_include_mask", None)
        
        # 🔍 디버그: force_include_mask 확인 (첫 100 스텝만)
        if self.use_pattern_masking and self.is_world_process_zero():
            step = getattr(self.state, "global_step", 0) or 0
            if step < 100 and step % 20 == 0:  # 처음 100 스텝 중 20마다
                if force_include_mask is not None:
                    mask_count = force_include_mask.sum().item()
                    total_count = force_include_mask.numel()
                    print(f"[DEBUG step={step}] force_include_mask: {mask_count}/{total_count} tokens protected")
                else:
                    print(f"[DEBUG step={step}] ⚠️  force_include_mask is None!")
        
        loss = self._profit_loss_from_logits(
            logits=logits, 
            labels=labels,
            force_include_mask=force_include_mask,
        )

        # ✅ 주기적 로깅: active ratio / counts
        if self.profit_log_every > 0 and self.is_world_process_zero():
            step = getattr(self.state, "global_step", 0) or 0
            if step > 0 and (step % self.profit_log_every == 0) and self._last_profit_stats:
                active = self._last_profit_stats.get("active_tokens", float("nan"))
                valid = self._last_profit_stats.get("valid_tokens", float("nan"))
                ratio = self._last_profit_stats.get("active_ratio", float("nan"))
                thr = self._last_profit_stats.get("threshold", float("nan"))
                forced = self._last_profit_stats.get("forced_tokens", 0.0)
                msg = (
                    f"[ProFit] step={step} threshold={thr:.4f} "
                    f"active={active:.0f} valid={valid:.0f} active_ratio={ratio:.4f}"
                )
                if forced > 0:
                    msg += f" forced={forced:.0f}"
                print(msg)

        return (loss, outputs) if return_outputs else loss

    def _profit_loss_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        force_include_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        logits: [B, T, V]
        labels: [B, T]
        force_include_mask: [B, T] (optional) - 정규식 패턴으로 생성된 마스크
        표준 causal LM CE 정렬: logits[:, :-1] vs labels[:, 1:]
        """
        logits = logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # force_include_mask도 shift
        shift_force_mask = None
        if force_include_mask is not None:
            shift_force_mask = force_include_mask[:, 1:].contiguous()

        _, _, V = shift_logits.shape
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1).to(flat_logits.device)
        flat_force_mask = shift_force_mask.view(-1) if shift_force_mask is not None else None

        return self._profit_cross_entropy_token_weighted(
            logits_2d=flat_logits,
            labels_1d=flat_labels,
            prob_threshold=self.prob_threshold,
            threshold_direction=self.threshold_direction,
            ignore_index=-100,
            force_include_mask_1d=flat_force_mask,
        )

    def _profit_cross_entropy_token_weighted(
        self,
        logits_2d: torch.Tensor,  # [N, V]
        labels_1d: torch.Tensor,  # [N]
        prob_threshold: list[float],
        threshold_direction: str,
        ignore_index: int = -100,
        force_include_mask_1d: Optional[torch.Tensor] = None,  # 🆕 정규식 패턴으로 생성된 마스크
    ) -> torch.Tensor:
        """
        1) log_softmax로 logp 계산
        2) 정답 토큰 logp만 gather
        3) p_correct = exp(logp_correct)
        4) detach(p_correct)로 마스킹 조건 산출
        5) active token들의 NLL을 sum
        6) (분산이면) loss_sum과 active_count를 all-reduce(sum)
        7) global_loss = global_loss_sum / global_active_count
        + (4) 로깅용 통계 저장
        + 🆕 강제 포함 토큰은 확률과 관계없이 항상 학습
        + 🆕 정규식 패턴으로 생성된 마스크 지원
        """
        # valid target mask (label != ignore_index)
        valid = labels_1d.ne(ignore_index)

        # gather를 위해 -100은 0으로 치환 (valid=False인 위치는 어차피 제외됨)
        gather_idx = labels_1d.clone()
        gather_idx[~valid] = 0
        gather_idx = gather_idx.clamp(min=0)

        # log probs
        log_probs = F.log_softmax(logits_2d, dim=-1)  # [N, V]
        logp_correct = log_probs.gather(dim=-1, index=gather_idx.unsqueeze(-1)).squeeze(-1)  # [N]
        p_correct = logp_correct.exp()  # [N] in (0,1]

        # stop-grad gate
        p_det = p_correct.detach()

        # masking condition
        if threshold_direction == "higher":
            threshold = float(prob_threshold[0])
            mask = p_det.lt(threshold)
        elif threshold_direction == "lower":
            threshold = float(prob_threshold[0])
            mask = p_det.gt(threshold)
        elif threshold_direction == "middle":
            low, high = prob_threshold
            threshold = float(low)  # 로깅용(대표값)
            mask = p_det.lt(low) | p_det.gt(high)
        elif threshold_direction == "random":
            threshold = float(prob_threshold[0])
            # (가독성 개선) valid인 위치에서만 random mask를 의미있게 적용
            mask = valid & torch.rand_like(p_det).lt(threshold)
        else:
            raise ValueError(f"Unknown threshold_direction: {threshold_direction}")

        # 🆕 강제 포함 토큰 마스크 생성 (토큰 ID 기반 + 정규식 패턴 기반)
        force_include_mask = torch.zeros_like(valid, dtype=torch.bool)

        # 1) 토큰 ID 기반 (기존 방식)
        if len(self._force_include_token_ids) > 0:
            for token_id in self._force_include_token_ids:
                force_include_mask |= labels_1d == token_id

        # 2) 정규식 패턴 기반 (새로운 방식)
        if force_include_mask_1d is not None:
            force_include_mask |= force_include_mask_1d.to(force_include_mask.device)

        # 최종 active = valid & ((~mask) | force_include)
        # 즉, 확률 기반 마스킹을 통과하거나 강제 포함 토큰이면 학습
        active = valid & ((~mask) | force_include_mask)

        # nll = -logp_correct (active만)
        nll = -logp_correct
        nll_sum = nll.masked_select(active).sum()  # scalar
        active_count = active.sum().to(dtype=nll_sum.dtype)  # scalar (float)
        valid_count = valid.sum().to(dtype=nll_sum.dtype)  # scalar (float)

        # 🆕 강제 포함된 토큰 수 계산
        forced_count = (valid & force_include_mask).sum().to(dtype=nll_sum.dtype)

        # 분산이면 전역 sum
        nll_sum_global = _all_reduce_sum_(nll_sum)
        active_count_global = _all_reduce_sum_(active_count)
        valid_count_global = _all_reduce_sum_(valid_count)
        forced_count_global = _all_reduce_sum_(forced_count)

        # divide (clamp to avoid zero)
        denom = torch.clamp(active_count_global, min=1.0)
        loss = nll_sum_global / denom

        # DeepSpeed 호환: 0-dim scalar 보장 (이미 scalar지만 명시적으로 처리)
        # multi-GPU에서 all_reduce 후에도 scalar shape 유지를 확실히 함
        # if loss.ndim != 0:
        # loss = loss.squeeze()

        # ✅ 로깅용 통계 저장(프로세스 0에서만 사용)
        # 값은 모든 rank에서 같아야 하므로 global 값을 씀
        with torch.no_grad():
            active_g = float(active_count_global.item())
            valid_g = float(valid_count_global.item())
            forced_g = float(forced_count_global.item())
            ratio_g = (active_g / valid_g) if valid_g > 0 else 0.0
            self._last_profit_stats = {
                "active_tokens": active_g,
                "valid_tokens": valid_g,
                "active_ratio": ratio_g,
                "threshold": float(threshold),
                "forced_tokens": forced_g,
            }

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
    force_include_tokens: Optional[list[str]] = None,
    force_include_patterns: Optional[list[str]] = None,
    use_pattern_masking: bool = False,
    **kwargs,
) -> ProFitSFTTrainer:
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

    return ProFitSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prob_threshold=prob_threshold,
        threshold_direction=threshold_direction,
        use_profit_loss=True,
        force_include_tokens=force_include_tokens,
        force_include_patterns=force_include_patterns,
        use_pattern_masking=use_pattern_masking,
    )


if __name__ == "__main__":
    print("ProFit SFT Trainer (patched)")
