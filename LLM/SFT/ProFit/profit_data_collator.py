"""
ProFit를 위한 Custom Data Collator

force_include_mask를 labels와 동일하게 처리합니다.
- padding 환경: 동일한 길이로 패딩
- padding_free 환경: flatten & concat
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


@dataclass
class ProFitDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    ProFit를 위한 Data Collator (Padding 환경용).
    
    기본 DataCollatorForLanguageModeling 기능 + force_include_mask padding 추가.
    """
    
    force_include_mask_pad_value: bool = False
    
    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        """배치 생성 및 padding."""
        # force_include_mask가 있는지 확인
        has_force_mask = "force_include_mask" in features[0]
        
        # force_include_mask 임시 제거 (부모 클래스가 처리 못함)
        force_mask_list = None
        if has_force_mask:
            force_mask_list = [f.pop("force_include_mask") for f in features]
        
        # 부모 클래스로 기본 padding (input_ids, labels, attention_mask)
        batch = super().__call__(features, return_tensors)
        
        # force_include_mask padding 및 추가
        if has_force_mask and force_mask_list is not None:
            max_length = batch["input_ids"].shape[1]
            batch_size = len(force_mask_list)
            
            padded_force_mask = torch.full(
                (batch_size, max_length),
                self.force_include_mask_pad_value,
                dtype=torch.bool,
            )
            
            for i, mask in enumerate(force_mask_list):
                if isinstance(mask, list):
                    mask = torch.tensor(mask, dtype=torch.bool)
                length = len(mask)
                padded_force_mask[i, :length] = mask
            
            batch["force_include_mask"] = padded_force_mask
            
            # features에 다시 추가 (원본 데이터 보존용)
            for i, f in enumerate(features):
                f["force_include_mask"] = force_mask_list[i]
        
        return batch


@dataclass
class ProFitDataCollatorPaddingFree:
    """
    ProFit를 위한 Padding-Free Data Collator.
    
    padding_free=True 환경에서 force_include_mask를 처리합니다.
    모든 시퀀스를 flatten하여 하나의 연속된 시퀀스로 만들 때,
    force_include_mask도 동일하게 concat합니다.
    """
    
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    
    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        """
        배치 생성 (padding-free).
        
        모든 샘플을 concat하여 하나의 긴 시퀀스로 만듭니다.
        """
        # force_include_mask가 있는지 확인
        has_force_mask = "force_include_mask" in features[0]
        
        # 각 필드를 concat
        batch = {}
        
        for key in ["input_ids", "labels", "attention_mask"]:
            if key in features[0]:
                tensors = []
                for f in features:
                    value = f[key]
                    if isinstance(value, list):
                        value = torch.tensor(value)
                    elif not isinstance(value, torch.Tensor):
                        value = torch.tensor(value)
                    tensors.append(value)
                
                # Flatten and concat, then add batch dimension [1, total_length]
                concatenated = torch.cat(tensors, dim=0)
                batch[key] = concatenated.unsqueeze(0)
        
        # force_include_mask도 동일하게 concat
        if has_force_mask:
            mask_tensors = []
            for f in features:
                mask = f["force_include_mask"]
                if isinstance(mask, list):
                    mask = torch.tensor(mask, dtype=torch.bool)
                elif not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, dtype=torch.bool)
                mask_tensors.append(mask)
            
            # Flatten and concat, then add batch dimension [1, total_length]
            concatenated_mask = torch.cat(mask_tensors, dim=0)
            batch["force_include_mask"] = concatenated_mask.unsqueeze(0)
        
        # position_ids 생성 (padding_free 환경용)
        position_ids = []
        for f in features:
            seq_len = len(f["input_ids"]) if isinstance(f["input_ids"], list) else f["input_ids"].size(0)
            position_ids.append(torch.arange(seq_len))
        # [1, total_length]
        batch["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
        
        # cu_seqlens (cumulative sequence lengths) - [num_sequences + 1]
        seq_lens = [len(f["input_ids"]) if isinstance(f["input_ids"], list) else f["input_ids"].size(0) 
                    for f in features]
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0)), dtype=torch.int32)
        batch["cu_seqlens"] = cu_seqlens
        
        # max_seqlen
        batch["max_seqlen"] = max(seq_lens)
        
        return batch


if __name__ == "__main__":
    print("ProFit Data Collator")
    print("="*70)
    print("force_include_mask를 labels와 동일하게 처리합니다.")
    print("\n두 가지 모드:")
    print("  1. ProFitDataCollatorForLanguageModeling - padding 환경용")
    print("  2. ProFitDataCollatorPaddingFree - padding_free=True 환경용")
    print("\n사용 예제:")
    print("""
# Padding 환경
collator = ProFitDataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Padding-Free 환경
collator = ProFitDataCollatorPaddingFree()

trainer = ProFitSFTTrainer(
    model=model,
    train_dataset=dataset_with_force_mask,
    data_collator=collator,  # ← 필수!
    use_pattern_masking=True,
)
    """)
    print("\n⚠️  주의: force_include_mask가 있는 데이터셋은 반드시 이 collator 사용!")
