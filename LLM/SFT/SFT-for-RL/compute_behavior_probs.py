"""
Behavior Policy (πβ) 확률값을 미리 계산하는 스크립트

PEAR 학습 시 reference model을 메모리에 올리지 않고,
미리 계산된 확률값을 사용하기 위해 offline으로 πβ를 계산합니다.

사용법:
    python compute_behavior_probs.py \
        --model_name_or_path "Qwen/Qwen2.5-1.5B" \
        --dataset_path "./data/train_dataset.jsonl" \
        --output_path "./data/train_dataset_with_probs.jsonl" \
        --batch_size 4

출력:
    원본 데이터셋 + behavior_probs (각 토큰의 log 확률값)
"""

import argparse
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

IGNORE_INDEX = -100


def compute_token_log_probs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    주어진 input_ids와 labels에 대해 각 토큰의 log 확률값을 계산.
    
    Args:
        model: Behavior policy model
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] (IGNORE_INDEX로 마스킹)
        device: 연산 device
    
    Returns:
        log_probs: [batch, seq_len] numpy array
            - 유효한 토큰: log p(yt | x, y<t)
            - 무효 토큰 (IGNORE_INDEX): 0.0
    """
    batch_size, seq_len = input_ids.shape
    
    # Move to device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    # Forward pass (no gradient)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits.float()  # [batch, seq_len, vocab]
    
    # Labels shift (다음 토큰 예측)
    labels_padded = F.pad(labels, (0, 1), value=IGNORE_INDEX)
    shift_labels = labels_padded[..., 1:].contiguous()
    
    # Log probabilities
    log_probs_all = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab]
    
    # 정답 토큰의 log 확률 추출
    labels_clamped = shift_labels.clamp(min=0)  # -100을 0으로 변환 (gather용)
    token_log_probs = log_probs_all.gather(
        dim=-1, index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)  # [batch, seq_len]
    
    # IGNORE_INDEX 위치는 0.0으로 설정
    valid_mask = (shift_labels != IGNORE_INDEX)
    token_log_probs = token_log_probs * valid_mask.float()
    
    # CPU로 이동 및 numpy 변환
    token_log_probs_np = token_log_probs.cpu().numpy()
    
    return token_log_probs_np


def collate_fn(batch):
    """
    배치 샘플들을 묶어서 텐서로 변환.
    가변 길이 시퀀스를 padding하여 동일 길이로 맞춤.
    """
    # input_ids와 labels 추출
    input_ids_list = [torch.tensor(item["input_ids"]) for item in batch]
    labels_list = [torch.tensor(item["labels"]) for item in batch]
    
    # Padding (오른쪽에 추가)
    max_len = max(len(ids) for ids in input_ids_list)
    
    input_ids_padded = []
    labels_padded = []
    
    for input_ids, labels in zip(input_ids_list, labels_list):
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=0)  # pad_token_id = 0
            labels = F.pad(labels, (0, pad_len), value=IGNORE_INDEX)
        
        input_ids_padded.append(input_ids)
        labels_padded.append(labels)
    
    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded),
        "original_batch": batch,  # 원본 데이터 보존
    }


def compute_behavior_probs_for_dataset(
    model_name_or_path: str,
    dataset: Dataset,
    batch_size: int = 4,
    torch_dtype: str = "auto",
    device: Optional[str] = None,
) -> Dataset:
    """
    데이터셋의 모든 샘플에 대해 behavior policy 확률값을 계산.
    
    Args:
        model_name_or_path: Behavior policy model path
        dataset: input_ids와 labels를 포함하는 데이터셋
        batch_size: 배치 크기
        torch_dtype: 모델 dtype
        device: 연산 device (None이면 자동 선택)
    
    Returns:
        원본 데이터셋 + behavior_log_probs 컬럼
    """
    # Device 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Loading behavior policy model: {model_name_or_path}")
    print(f"Device: {device}")
    
    # Model 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype if torch_dtype != "auto" else "auto",
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # 단순화를 위해 0으로 설정
    )
    
    # 결과 저장용 리스트
    all_log_probs = []
    
    print(f"Computing behavior policy probabilities for {len(dataset)} samples...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        original_batch = batch["original_batch"]
        
        # Log probabilities 계산
        log_probs_batch = compute_token_log_probs(model, input_ids, labels, device)
        
        # 각 샘플별로 원래 길이만큼만 추출 (padding 제거)
        for i, sample in enumerate(original_batch):
            original_len = len(sample["input_ids"])
            sample_log_probs = log_probs_batch[i, :original_len]
            all_log_probs.append(sample_log_probs.tolist())
    
    # 데이터셋에 컬럼 추가
    print("Adding behavior_log_probs column to dataset...")
    dataset = dataset.add_column("behavior_log_probs", all_log_probs)
    
    print(f"✓ Behavior policy probabilities computed for {len(dataset)} samples")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Compute behavior policy probabilities")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Behavior policy model path (데이터를 생성한 모델)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset path (input_ids, labels를 포함해야 함)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output dataset path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for computation",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="json",
        choices=["json", "jsonl", "csv", "parquet"],
        help="Dataset format",
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 데이터셋 로드
    print(f"Loading dataset from: {args.dataset_path}")
    if args.dataset_format == "jsonl":
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    elif args.dataset_format == "json":
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    elif args.dataset_format == "csv":
        dataset = load_dataset("csv", data_files=args.dataset_path, split="train")
    elif args.dataset_format == "parquet":
        dataset = load_dataset("parquet", data_files=args.dataset_path, split="train")
    else:
        raise ValueError(f"Unsupported dataset format: {args.dataset_format}")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")
    
    # input_ids와 labels 체크
    if "input_ids" not in dataset.features or "labels" not in dataset.features:
        raise ValueError(
            "Dataset must contain 'input_ids' and 'labels' columns. "
            "Please preprocess your dataset first."
        )
    
    # Behavior policy 확률 계산
    dataset_with_probs = compute_behavior_probs_for_dataset(
        model_name_or_path=args.model_name_or_path,
        dataset=dataset,
        batch_size=args.batch_size,
        torch_dtype=args.torch_dtype,
        device=args.device,
    )
    
    # 결과 저장
    print(f"Saving dataset with behavior_log_probs to: {args.output_path}")
    
    # 출력 형식에 따라 저장
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.output_path.endswith(".jsonl") or args.output_path.endswith(".json"):
        dataset_with_probs.to_json(args.output_path, orient="records", lines=True)
    elif args.output_path.endswith(".csv"):
        dataset_with_probs.to_csv(args.output_path, index=False)
    elif args.output_path.endswith(".parquet"):
        dataset_with_probs.to_parquet(args.output_path)
    else:
        # 기본값: JSON Lines
        dataset_with_probs.to_json(args.output_path, orient="records", lines=True)
    
    print("✓ Complete!")
    print(f"  - Input samples: {len(dataset)}")
    print(f"  - Output samples: {len(dataset_with_probs)}")
    print(f"  - New column: behavior_log_probs")
    print(f"  - Output path: {args.output_path}")


if __name__ == "__main__":
    main()
