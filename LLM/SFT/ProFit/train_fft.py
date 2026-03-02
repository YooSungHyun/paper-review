import argparse
import os
from typing import Optional
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from accelerate import PartialState, logging
from datasets import Dataset
from torch.utils.data.dataloader import _utils
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from arguments.ModelArguments import AILABModelConfig
from utils import prepare_exp, read_txt, write_txt
from utils.DataProcessor import DirectProcessor
from utils.MarkdownParser import wrap_with_backticks
from utils.masking_utils import RAFT_PATTERNS
from profit_sft_trainer import ProFitSFTTrainer
from transformers.trainer_utils import is_main_process

_utils.MP_STATUS_CHECK_INTERVAL = 600.0
torch._dynamo.config.verbose = True
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IGNORE_INDEX = -100
logger = logging.get_logger(__name__)


def dft_token_level_loss(outputs, labels, num_items_in_batch=None):
    """
    DFT 논문 식 (9) 구현: per-token loss = (-log p_t) * stop_grad(p_t)
    - logits: (B, T, V)
    - labels: (B, T)  (IGNORE_INDEX로 마스킹된 위치 포함)
    - padding_free(packed) 환경에서도 동작 (경계/패딩은 labels가 IGNORE로 들어와야 함)

    Returns:
        loss: 스칼라 (유효 토큰 평균)
    """
    logits = outputs.logits

    # Teacher-forcing shift: 토큰 t가 다음 토큰 t+1을 예측
    # (라벨만 한 칸 오른쪽으로 밀고 마지막을 IGNORE로 만드는 전형적 방식)
    labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)
    shift_labels = labels[..., 1:].contiguous()  # (B, T)
    B, T, V = logits.shape

    # (B*T, V), (B*T,)
    logits = logits.reshape(-1, V)
    shift_labels = shift_labels.reshape(-1).to(logits.device)

    # 유효 토큰만 사용
    valid_mask = shift_labels != IGNORE_INDEX
    if not torch.any(valid_mask):
        # 모든 토큰이 IGNORE인 배치면 0으로 리턴
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    valid_logits = logits[valid_mask]  # (N, V)
    valid_labels = shift_labels[valid_mask]  # (N,)

    # CE per-token = -log p_t
    # (log_softmax를 따로 써도 동일하지만 CE가 내부적으로 사용함)
    token_ce = F.cross_entropy(valid_logits, valid_labels, reduction="none")  # (N,)

    # 목표 토큰 확률 p_t (stop-grad)
    with torch.no_grad():
        log_probs = F.log_softmax(valid_logits, dim=-1)  # (N, V)
        target_log_probs = log_probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1)  # (N,)
        p_t = target_log_probs.exp().clamp_(min=1e-8)  # (N,)

    # DFT: (-log p_t) * p_t
    weighted = token_ce * p_t.detach()

    # 유효 토큰 평균으로 정규화 (packed 환경에서도 길이 편차에 안전)
    loss = weighted.mean()
    return loss


def load_train_dataset() -> Dataset:
    def balance_answers(df, column="answer", keyword="@@@@", ratio=0.25):
        # 1. 그룹 나누기
        group_with = df[df[column].str.contains(keyword, na=False)]
        group_without = df[~df[column].str.contains(keyword, na=False)]

        # 2. 개수 기준 계산
        count_without = len(group_without)
        sample_count_with = int(count_without * ratio)

        # 3. 샘플링
        group_with_sampled = (
            group_with.sample(n=sample_count_with, random_state=42)
            if sample_count_with < len(group_with)
            else group_with
        )

        # 4. 병합
        final_df = pd.concat([group_without, group_with_sampled], ignore_index=True)

        # 5. 셔플 (선택)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return final_df

    def normalize_markdown_bullets(text: str) -> str:
        """
        마크다운 텍스트에서 *, + 리스트 기호를 모두 - 로 통일하되
        들여쓰기 레벨 및 서브리스트는 유지.
        """
        # ^ : 줄의 시작
        # (\s*) : 앞의 공백 (들여쓰기)
        # [\*\+] : * 또는 + 기호
        # (?=\s) : 그 뒤에 반드시 공백이 있는 경우만
        pattern = r"^(\s*)[\*\+](?=\s)"
        return re.sub(pattern, r"\1-", text, flags=re.MULTILINE)

    # human_df = pd.read_csv("./data/train_human (11)_shuffled_no_agree_revision (v0.2.hotfix3.1).csv", index_col=False)
    # human_df = pd.read_csv("./data/train_human (11)_shuffled_no_agree.csv", index_col=False)
    human_df = pd.read_csv("./data/train_human (15)_shuffled.csv", index_col=False)

    normal_df = human_df[human_df["비고"].isin(["yj100 스타일", "실제 고객 스타일"])]
    edge_df = human_df[human_df["비고"].isin(["context O, 답변 X", "context X, 답변 X"])]
    normal_df = normal_df.replace("", np.nan).dropna(
        subset=["question", "context1", "context2", "context3", "정답context", "reason", "answer"]
    )
    edge_df[edge_df["비고"] == "context X, 답변 X"].loc[:, ["context1", "context2", "context3", "reason"]] = edge_df[
        ["context1", "context2", "context3", "reason"]
    ].where(edge_df[["context1", "context2", "context3", "reason"]].isna(), np.nan)
    edge_df.fillna("", inplace=True)

    # reason in answer 학습하려면 주석해제
    # normal_df = normal_df[normal_df["answer"].str.contains("@@@@")]

    normal_df["reason"] = normal_df["reason"].apply(wrap_with_backticks)
    normal_df["answer"] = normal_df["answer"].apply(normalize_markdown_bullets)
    # balanced_df = balance_answers(normal_df)
    final_df = pd.concat([edge_df, normal_df], ignore_index=True)
    human_dataset = Dataset.from_pandas(final_df)

    return human_dataset


def main(script_args: ScriptArguments, training_args: SFTConfig, model_args: AILABModelConfig) -> None:

    # (1) Preparing experiment
    prepare_exp("pack_test", training_args)

    # (2) Load prompt templates
    raft_sys_prompt: str = read_txt("./prompt_template/raft/direct/system.txt")
    raft_user_prompt: str = read_txt("./prompt_template/raft/direct/user.txt")
    raft_label_prompt: str = read_txt("./prompt_template/raft/direct/label.txt")

    # (3) Load trainable model
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=model_args.padding_side,
        use_fast=True,
    )

    # (4) Load and preprocess train dataset
    with PartialState().local_main_process_first():
        train_dataset = load_train_dataset()
        preprocessor = DirectProcessor()

        train_dataset = train_dataset.map(
            preprocessor.preprocess_fft,
            fn_kwargs={
                "tokenizer": tokenizer,
                "system_prompt": raft_sys_prompt,
                "user_prompt_template": raft_user_prompt,
                "label_prompt": raft_label_prompt,
                "ignore_idx": IGNORE_INDEX,
            },
            remove_columns=list(train_dataset.features),
            num_proc=training_args.dataset_num_proc,
        )
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= training_args.max_length)

    # 해당 옵션은 없어도, input_ids가 있으면, SFTTrainer에서 자동으로 무시되나, 옵션이 있으면, 굳이 펑션내부로 스택트레이스 들어갈 필요없이 바로 넘어감.
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # (5) Run training
    # 🆕 정규식 패턴 기반 마스킹 설정
    use_pattern_masking = True  # True: 정규식으로 처리, False: 일반 텍스트로 처리

    if use_pattern_masking:
        # 정규식 패턴 모드: force_include_patterns에 정규식 패턴 리스트 전달
        trainer = ProFitSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            prob_threshold=model_args.prob_threshold,
            threshold_direction=model_args.threshold_direction,
            use_profit_loss=model_args.use_profit_loss,
            profit_log_every=model_args.profit_log_every,
            force_include_patterns=RAFT_PATTERNS + [re.escape(tok) for tok in tokenizer.all_special_tokens],
            use_pattern_masking=True,
        )
    else:
        # 일반 텍스트 모드: force_include_tokens에 토큰 텍스트 리스트 전달
        trainer = ProFitSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            prob_threshold=model_args.prob_threshold,
            threshold_direction=model_args.threshold_direction,
            use_profit_loss=model_args.use_profit_loss,
            profit_log_every=model_args.profit_log_every,
            force_include_tokens=[
                "<title: ",
                ">",
                "</title>",
                "<content: ",
                "</content>",
                "<question>",
                "</question>",
                "<answer>",
                "</answer>",
            ]
            + tokenizer.all_special_tokens,
            use_pattern_masking=False,
        )

    trainer.train()

    # (6) Save outputs
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    write_txt(os.path.join(training_args.output_dir, "raft_system.txt"), raft_sys_prompt)
    write_txt(os.path.join(training_args.output_dir, "raft_user.txt"), raft_user_prompt)
    write_txt(os.path.join(training_args.output_dir, "raft_label.txt"), raft_label_prompt)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, AILABModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
