"""
정규식 기반 토큰 마스킹 유틸리티

특정 패턴에 매칭되는 텍스트 구간의 토큰을 마스킹하지 않도록 처리합니다.
"""
import re
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizerBase


def find_pattern_char_spans(text: str, patterns: List[str]) -> List[Tuple[int, int]]:
    """
    텍스트에서 정규식 패턴에 매칭되는 모든 구간의 character span을 찾습니다.
    
    Args:
        text: 입력 텍스트
        patterns: 정규식 패턴 리스트
        
    Returns:
        [(start, end), ...] 형태의 character span 리스트
    """
    spans = []
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            spans.append((match.start(), match.end()))
    
    # 겹치는 구간 병합
    if not spans:
        return []
    
    spans.sort()
    merged = [spans[0]]
    
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # 겹치면 병합
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged


def char_spans_to_token_indices(
    text: str,
    char_spans: List[Tuple[int, int]],
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = False,
) -> List[int]:
    """
    Character span을 토큰 인덱스로 변환합니다.
    
    Args:
        text: 원본 텍스트
        char_spans: Character span 리스트 [(start, end), ...]
        tokenizer: 토크나이저
        add_special_tokens: 특수 토큰 포함 여부
        
    Returns:
        토큰 인덱스 리스트 (중복 제거됨)
    """
    if not char_spans:
        return []
    
    # 토크나이징
    encoding = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    
    token_indices = set()
    offsets = encoding["offset_mapping"]
    
    # 각 character span에 대해 겹치는 토큰 찾기
    for char_start, char_end in char_spans:
        for token_idx, (token_start, token_end) in enumerate(offsets):
            # 토큰과 character span이 겹치는지 확인
            if token_start < char_end and token_end > char_start:
                token_indices.add(token_idx)
    
    return sorted(token_indices)


def create_force_include_mask(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    patterns: List[str],
    add_special_tokens: bool = False,
) -> List[bool]:
    """
    정규식 패턴에 매칭되는 구간의 토큰을 True로 표시하는 마스크를 생성합니다.
    
    Args:
        text: 입력 텍스트
        tokenizer: 토크나이저
        patterns: 정규식 패턴 리스트
        add_special_tokens: 특수 토큰 포함 여부
        
    Returns:
        각 토큰마다 force_include 여부를 나타내는 bool 리스트
    """
    # 패턴 매칭
    char_spans = find_pattern_char_spans(text, patterns)
    
    if not char_spans:
        # 매칭 없음 -> 모두 False
        encoding = tokenizer(text, add_special_tokens=add_special_tokens)
        return [False] * len(encoding["input_ids"])
    
    # 토큰 인덱스 변환
    force_include_indices = char_spans_to_token_indices(
        text, char_spans, tokenizer, add_special_tokens
    )
    
    # 마스크 생성
    encoding = tokenizer(text, add_special_tokens=add_special_tokens)
    num_tokens = len(encoding["input_ids"])
    
    mask = [False] * num_tokens
    for idx in force_include_indices:
        if idx < num_tokens:
            mask[idx] = True
    
    return mask


# 자주 사용하는 패턴 프리셋
RAFT_PATTERNS = [
    r"<title:\s*\S+>.*?</title>",  # <title: doc_id>...</title> (텍스트 ID)
    r"<content:\s*\S+>.*?</content>",  # <content: doc_id>...</content> (텍스트 ID)
    r"<question>.*?</question>",  # <question>...</question>
    r"<answer>",  # <answer> 태그만
    r"</answer>",  # </answer> 태그만
]


if __name__ == "__main__":
    # 테스트
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    
    test_text = """<title: 0>제 2 관 보험금의 지급</title>에 따르면, <content: 0>이 계약에서 계약자가 보험수익자를 지정하지 않은 때에는</content>"""
    
    print("=" * 70)
    print("정규식 기반 토큰 마스킹 유틸리티 테스트")
    print("=" * 70)
    
    # Character span 찾기
    char_spans = find_pattern_char_spans(test_text, RAFT_PATTERNS)
    print(f"\n매칭된 character spans: {char_spans}")
    
    for start, end in char_spans:
        print(f"  - [{start}:{end}] = '{test_text[start:end]}'")
    
    # 토큰 인덱스 변환
    token_indices = char_spans_to_token_indices(test_text, char_spans, tokenizer)
    print(f"\n매칭된 토큰 인덱스: {token_indices}")
    
    # 마스크 생성
    mask = create_force_include_mask(test_text, tokenizer, RAFT_PATTERNS)
    
    # 토큰별로 출력
    encoding = tokenizer(test_text)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    
    print(f"\n토큰별 force_include 상태:")
    print("-" * 70)
    for idx, (token, include) in enumerate(zip(tokens, mask)):
        mark = "✓" if include else " "
        print(f"  [{mark}] {idx:3d}: {token}")
    
    print("=" * 70)
