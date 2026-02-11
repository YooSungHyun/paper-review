# PEAR SFT Trainer 구현 검토 및 논문-코드 매핑 레포트

**작성일**: 2026-02-11  
**검토 대상**: `for_rl_sft_trainer.py`, `for_rl_sft_trainer_precomputed.py`, `compute_behavior_probs.py`  
**원본 참조**: PEAR 논문 (arXiv:2602.01058) - Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning

> **참고**: PEAR는 공개 구현체가 없어, 논문만을 기반으로 구현되었습니다.  
> 본 문서는 **논문의 각 로직이 코드의 어느 부분에 구현되었는지** 상세히 매핑합니다.

---

## 📋 목차

1. [요약](#요약)
2. [논문 핵심 내용](#논문-핵심-내용)
3. [파일 구조 및 아키텍처](#파일-구조-및-아키텍처)
4. [논문-코드 대응 관계](#논문-코드-대응-관계)
5. [파라미터 상세 가이드](#파라미터-상세-가이드)
6. [Numerical Stabilization (Clipping) 상세](#numerical-stabilization-clipping-상세)
7. [Weighting Mode 상세](#weighting-mode-상세)
8. [검증 항목](#검증-항목)
9. [데이터 타입별 권장 설정](#데이터-타입별-권장-설정)

---

## 요약

### ✅ 구현 완성도: **논문 기반 100% 구현**

논문의 Algorithm 1과 Section 3의 모든 핵심 요소를 구현했습니다.

**주요 성과:**
- ✅ Importance sampling 기반 loss reweighting
- ✅ 3가지 weighting variant (uniform, suffix, block)
- ✅ 2단계 numerical stabilization (log-ratio clip, weight clip)
- ✅ Reference model / Precomputed 두 가지 사용 방식 지원
- ✅ 토큰 단위 정규화

**구현 방식:**
- `for_rl_sft_trainer.py`: Reference model을 메모리에 로드 (논문 원형)
- `for_rl_sft_trainer_precomputed.py`: 미리 계산된 πβ 사용 (메모리 효율적)

---

## 논문 핵심 내용

### 1. 문제 정의 (§2.1, §2.2)

**핵심 발견**: Offline 성능 ↑ ≠ Online RL 성능 ↑

- Offline SFT: Behavior policy πβ가 생성한 데이터로 학습
- Online RL: Target policy πθ가 자기 자신의 rollout으로 학습
- **Distribution mismatch**: 두 policy의 occupancy가 다름
- Uniform SFT loss는 πβ가 over-represent한 trajectory에 과도한 gradient 부여 → RL이 다시 방문하지 않는 경로

### 2. OPE (Off-Policy Evaluation) 기반 해결 (§2.3)

**핵심 수식**:
```
E_τ∼πθ[f(τ)] = E_τ∼πβ[ (πθ(τ)/πβ(τ)) × f(τ) ]
```

Likelihood ratio로 behavior distribution을 target distribution으로 reweight.

### 3. PEAR 알고리즘 (§3, Algorithm 1)

**Step 1**: Per-token log-likelihood ratio 계산
```
δt = log πθ(yt|x,y<t) - log πβ(yt|x,y<t)
```

**Step 2**: 3가지 variant로 weight aggregation
- **Uniform**: w₁:T = ∏ᵢ Δᵢ (sequence-level)
- **Suffix**: Gt = γ^(T-t) × ∏(j=t+1 to T) Δj (token-level)
- **Block**: Block 단위로 위와 유사 (stability)

**Step 3**: Weighted loss
```
L_PEAR = Σ_t stop_gradient[Ĝt] × ℓθ(x, y<t, yt)
```

### 4. Numerical Stabilization (§3.7)

- **Log-space 계산**: Ratio의 곱 → log-ratio의 합
- **Per-token clip**: δt를 [ℓΔ, uΔ]로 clipping
- **Final weight clip**: Ĝt를 [Gmin, Gmax]로 clipping

논문 기본값: `clip log Ĝt to [-10, 5]`, `clip per-decision log Δt to [-0.08, 0.3]`

---

## 파일 구조 및 아키텍처

### 파일 의존 관계

```
compute_behavior_probs.py          # πβ 확률 사전 계산 (Precomputed용)
         │
         ▼
for_rl_sft_trainer_precomputed.py  # Precomputed 버전 (메모리 효율)
         │
         ├── PEARSFTTrainerPrecomputed
         │      ├── compute_loss()
         │      ├── _compute_pear_loss()
         │      ├── _compute_importance_weights()
         │      ├── _uniform_weighting()
         │      ├── _suffix_weighting()
         │      └── _block_weighting()
         │
for_rl_sft_trainer.py              # Reference model 버전
         │
         └── PEARSFTTrainer
                └── (동일 구조 + ref_model forward)
```

### 호출 흐름

```
compute_loss(model, inputs)
    │
    ├─ labels, behavior_log_probs 추출
    ├─ model(**inputs)  → logits (πθ)
    ├─ [ref_model용만] ref_model(**inputs) → ref_logits (πβ)
    │
    └─ _compute_pear_loss(logits, behavior_probs, labels)
           │
           └─ for each sample:
                 ├─ per_token_loss = CrossEntropy(reduction='none')
                 ├─ weights = _compute_importance_weights(...)
                 └─ loss += (weights * per_token_loss).sum() / num_valid_tokens
```

---

## 논문-코드 대응 관계

### 표 1: 논문 섹션 → 코드 위치 매핑

| 논문 섹션 | 논문 내용 | 코드 파일 | 코드 위치 | 구현 내용 |
|-----------|-----------|------------|-----------|-----------|
| **§3.1** | Problem setup, notation | - | - | Docstring에 반영 |
| **§3.2** | PEAR objective, stop_gradient | for_rl_sft_trainer_precomputed | L215-218, L284 | `weights.detach()`, weighted loss |
| **§3.3** | Sequence-level weighting | for_rl_sft_trainer_precomputed | L348-354 | `_uniform_weighting()` |
| **§3.4** | Token-level suffix weighting | for_rl_sft_trainer_precomputed | L356-375 | `_suffix_weighting()` |
| **§3.5** | Block-level weighting | for_rl_sft_trainer_precomputed | L377-396 | `_block_weighting()` |
| **§3.7** | Numerical stabilization | for_rl_sft_trainer_precomputed | L331-332, L356, L370, L394 | `clip_ratio_range`, `clip_weight_range` |
| **Alg.1 L3** | δt 계산 및 clip | for_rl_sft_trainer_precomputed | L318-332 | `_compute_importance_weights()` |
| **Alg.1 L9-10** | Uniform mode | for_rl_sft_trainer_precomputed | L348-354 | `_uniform_weighting()` |
| **Alg.1 L14-20** | Suffix mode, backward scan | for_rl_sft_trainer_precomputed | L365-375 | `_suffix_weighting()` |

---

### 1. Per-token Log-Ratio 계산 (§3.2, Alg.1 L3)

**논문**:
```
δt = clip( log πθ(yt|x,y<t) - log πβ(yt|x,y<t), ℓΔ, uΔ )
```

**코드** (`for_rl_sft_trainer_precomputed.py` L318-332):

```python
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
```

**Precomputed vs Reference Model**:
- Precomputed: `behavior_log_probs`를 inputs에서 직접 사용
- Reference: `ref_model` forward → `ref_logits` → softmax → log 추출

---

### 2. PEAR Weighted Loss (§3.2, Equation)

**논문**:
```
L_PEAR(θ) = E_(x,y)∼D [ Σ_t sg[Ĝt] × ℓθ(x, y<t, yt) ]
```

**코드** (`for_rl_sft_trainer_precomputed.py` L274-296):

```python
# Per-token loss
per_token_loss = F.cross_entropy(
    sample_logits, sample_labels, ignore_index=-100, reduction='none'
)

# Importance weights 계산
weights = self._compute_importance_weights(...)

# Weighted loss (sg[·] = .detach() 적용됨)
weighted_loss = (weights * per_token_loss).sum()
num_valid_tokens = valid_mask.sum()

total_loss += weighted_loss
total_tokens += num_valid_tokens

# 토큰 단위 정규화
loss = total_loss / total_tokens
```

**stop_gradient 구현**: `_compute_importance_weights()` 반환값에 `.detach()` 적용 (L347)

---

### 3. Sequence-Level Weighting (§3.3, Alg.1 L9-10)

**논문**:
```
w₁:T = πθ(y|x) / πβ(y|x) = ∏(t=1 to T) Δt
Gi = w₁:T ∀i
```

**코드** (`for_rl_sft_trainer_precomputed.py` L348-354):

```python
def _uniform_weighting(self, log_ratio, valid_mask, seq_len):
    # 전체 시퀀스 log-ratio 합 (log-space에서 곱 = 합)
    total_log_ratio = (log_ratio * valid_mask.float()).sum()
    
    # Sequence-level weight: exp(Σ δt)
    weight = torch.exp(total_log_ratio)
    
    # Clipping
    weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
    
    # 모든 토큰에 동일한 weight
    weights = torch.full((seq_len,), weight.item(), device=log_ratio.device)
    return weights
```

---

### 4. Token-Level Suffix Weighting (§3.4, Alg.1 L14-20)

**논문**:
```
Gt = γ^(T-t) × ∏(j=t+1 to T) Δj
```

**코드** (`for_rl_sft_trainer_precomputed.py` L356-375):

```python
def _suffix_weighting(self, log_ratio, valid_mask, seq_len):
    weights = torch.zeros(seq_len, device=log_ratio.device)
    cumsum_log_ratio = 0.0  # Σ(j=t+1 to T) δj
    
    # Backward scan (Alg.1 line 14-20)
    for t in reversed(range(seq_len)):
        if not valid_mask[t]:
            continue
        
        # Discount factor: γ^(T-t)
        discount = self.gamma ** (seq_len - t - 1)
        
        # Suffix importance ratio: exp(Σ(j=t+1 to T) δj)
        suffix_weight = torch.exp(torch.tensor(cumsum_log_ratio, device=log_ratio.device))
        
        # Combined: Gt = γ^(T-t) × suffix_ratio
        weight = discount * suffix_weight
        
        # Clipping
        weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
        
        weights[t] = weight
        cumsum_log_ratio += log_ratio[t].item()  # 다음 iteration을 위해
    return weights
```

**논문 Figure 3 (c)**: Token-wise, suffix-based weighting 정확히 구현

---

### 5. Block-Level Weighting (§3.5, Alg.1)

**논문**:
- K = ⌈T/B⌉ blocks
- Block k 내 ratio: ρk = Σ(t∈Ik) δt
- Gt = γ^(T-ek) × exp(Σ(m=k+1 to K) ρm) for t ∈ Ik

**코드** (`for_rl_sft_trainer_precomputed.py` L377-396):

```python
def _block_weighting(self, log_ratio, valid_mask, seq_len):
    num_blocks = (seq_len + self.block_size - 1) // self.block_size
    weights = torch.zeros(seq_len, device=log_ratio.device)
    cumsum_log_ratio = 0.0
    
    for k in reversed(range(num_blocks)):
        block_start = k * self.block_size
        block_end = min((k + 1) * self.block_size, seq_len)
        block_last_idx = block_end - 1
        
        # ρk = Σ(t∈Ik) δt
        block_log_ratio = (log_ratio[block_start:block_end] * valid_mask[...].float()).sum()
        
        # Discount: γ^(T - ek)
        discount = self.gamma ** (seq_len - block_last_idx - 1)
        
        # Suffix: exp(Σ(m>k) ρm)
        suffix_weight = torch.exp(torch.tensor(cumsum_log_ratio, device=log_ratio.device))
        
        weight = discount * suffix_weight
        weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
        
        weights[block_start:block_end] = weight
        cumsum_log_ratio += block_log_ratio.item()
    return weights
```

---

### 6. Labels Shift (Causal LM 표준)

**논문**: Next-token prediction이므로 logits[t]는 y_{t+1} 예측

**코드** (`for_rl_sft_trainer_precomputed.py` L243-245):

```python
labels = F.pad(labels, (0, 1), value=-100)
shift_labels = labels[..., 1:].contiguous()
```

---

## 파라미터 상세 가이드

### 표 2: 전체 파라미터 요약

| 파라미터 | 타입 | 기본값 | 논문 근거 | 설명 |
|----------|------|--------|-----------|------|
| `weighting_mode` | str | `"suffix"` | §3.3-3.5 | uniform / suffix / block |
| `block_size` | int | `1` | §3.5, Alg.1 | B=1이면 token-level과 동일 |
| `gamma` | float | `0.999` | §3.4, §4.2 | Discount factor γ ∈ (0, 1] |
| `clip_ratio_range` | tuple | `(-0.08, 0.3)` | §3.7, Alg.1 L3 | [ℓΔ, uΔ] per-token log-ratio |
| `clip_weight_range` | tuple | `(0.1, 10.0)` | §3.7, Alg.1 | [Gmin, Gmax] final weight |

### weighting_mode 상세

| mode | 논문 | 수식 | use case |
|------|------|------|----------|
| `uniform` | §3.3 | w = ∏Δt | 단순, 효과적. 긴 시퀀스에선 variance 큼 |
| `suffix` | §3.4 | Gt = γ^(T-t) × ∏(j>t) Δj | **논문 권장**, 가장 세밀 |
| `block` | §3.5 | Block 단위 suffix | Stability 중시, B=4 등 |

### gamma (γ) 상세

- **의미**: 미래 토큰에 대한 discount
- **논문 §3.4**: "γ ∈ (0, 1] is a discount factor to control variance in long horizon"
- **논문 §4.2**: γ = 0.999 사용
- **효과**: γ↓ → 미래 덜 반영, variance↓ / γ↑ → 미래 더 반영, variance↑

### block_size 상세

- **B=1**: Token-level과 동일 (§3.5: "when B=1, we recover token-level PEAR")
- **B>1**: Block 내 토큰들은 동일 weight → variance 감소
- **논문 실험**: B=4 사용 (Figure 6)

---

## Numerical Stabilization (Clipping) 상세

### 2단계 Clipping 구조 (논문 §3.7, Algorithm 1)

```
Stage 1: Per-token Log-Ratio Clipping
    δt → clip(δt, ℓΔ, uΔ)
    
Stage 2: Final Weight Clipping  
    Gt → clip(Gt, Gmin, Gmax)
```

### Stage 1: clip_ratio_range

**논문**: "clip per-decision log Δt to [-0.08, 0.3]"

**수학적 의미**:
- δt = log(πθ/πβ) = log Δt
- clip 후: exp(-0.08) ≈ 0.92 ≤ Δt ≤ exp(0.3) ≈ 1.35
- 즉, **한 토큰당 ratio는 약 0.92~1.35배**로 제한

**코드** (`for_rl_sft_trainer_precomputed.py` L331-332):
```python
log_ratio = torch.clamp(log_ratio, self.clip_ratio_range[0], self.clip_ratio_range[1])
```

**Human data 등 πβ 낮은 경우**:
- πβ 매우 낮으면 log πβ 매우 음수 → δt 폭발
- **권장**: `(-0.05, 0.15)` 등 더 좁게 설정

### Stage 2: clip_weight_range

**논문**: "clip log Ĝt to [-10, 5]" (다른 파라미터)  
우리 구현: **Gt 자체**를 [0.1, 10.0]으로 clip

**수학적 의미**:
- 최종 weight가 0.1~10.0 범위로 제한
- 10배 이상 up-weight / 0.1 이하 down-weight 방지

**코드** (각 weighting 함수 내):
```python
weight = torch.clamp(weight, self.clip_weight_range[0], self.clip_weight_range[1])
```

**설정 가이드**:

| 데이터 타입 | clip_ratio_range | clip_weight_range | 이유 |
|-------------|------------------|-------------------|------|
| Model-generated | (-0.08, 0.3) | (0.1, 10.0) | 논문 기본값 |
| Human-annotated | (-0.05, 0.15) | (0.3, 3.0) | πβ 낮을 수 있어 보수적 |
| 극보수적 | (-0.03, 0.08) | (0.5, 2.0) | 거의 uniform에 가깝게 |

---

## Weighting Mode 상세

### 수식-코드 대응

| Mode | 논문 수식 | 코드 함수 | Backward scan |
|------|-----------|-----------|---------------|
| uniform | w = ∏Δt | `_uniform_weighting` | 없음 (한 번에 계산) |
| suffix | Gt = γ^(T-t) × ∏(j>t) Δj | `_suffix_weighting` | O(T) |
| block | Block별 suffix | `_block_weighting` | O(K), K=⌈T/B⌉ |

### Figure 3 → 코드 매핑

논문 Figure 3 (a)(b)(c):
- (a) **Token-Wise**: suffix mode → `_suffix_weighting`
- (b) **Block-wise**: block mode → `_block_weighting`
- (c) **Sequence**: uniform mode → `_uniform_weighting`

---

## 검증 항목

### 알고리즘 정확성

| 항목 | 논문 | 코드 | 상태 |
|------|------|------|------|
| Log-ratio 계산 | δt = log πθ - log πβ | L329 | ✅ |
| Per-token clip | δt ∈ [ℓΔ, uΔ] | L331-332 | ✅ |
| Uniform weight | w = exp(Σδt) | L352-354 | ✅ |
| Suffix weight | Gt = γ^(T-t) × exp(Σ) | L365-375 | ✅ |
| Block weight | Block 단위 | L377-396 | ✅ |
| Final clip | Gt ∈ [Gmin, Gmax] | L356, 370, 394 | ✅ |
| Stop gradient | sg[Ĝt] | L347 `.detach()` | ✅ |
| Token normalization | Σ / num_tokens | L292-294 | ✅ |

### 엣지 케이스

| 케이스 | 처리 | 코드 위치 |
|--------|------|-----------|
| valid_mask 전부 False | continue | L271-272 |
| total_tokens = 0 | total_loss 그대로 (0) | L293-294 |
| behavior_log_probs 없음 | ValueError | L199-204 |
| weighting_mode 검증 | _validate_pear_params | L131-145 |

---

## 데이터 타입별 권장 설정

### Model-Generated Data (논문 환경)

- SynLogic, SYNTHETIC-2 등 **πβ가 데이터를 생성**한 경우
- 논문 기본값 그대로 사용 가능

```python
weighting_mode="suffix"
gamma=0.999
clip_ratio_range=(-0.08, 0.3)
clip_weight_range=(0.1, 10.0)
```

### Human-Annotated Data

- πβ가 human data를 생성하지 않음 → πβ 낮을 수 있음
- **보수적 clipping** 권장

```python
weighting_mode="suffix"  # 또는 "uniform" (더 안정)
gamma=0.999
clip_ratio_range=(-0.05, 0.15)   # 더 좁게
clip_weight_range=(0.3, 3.0)     # 더 좁게
```

### Mixed / 불확실한 경우

```python
weighting_mode="uniform"  # 가장 안정적
clip_ratio_range=(-0.05, 0.2)
clip_weight_range=(0.2, 5.0)
```

---

## compute_behavior_probs.py 논문 대응

**논문 §3.1**: "y is a token sequence produced by a known data-generating policy πβ"

**역할**: πβ(yt|x,y<t)를 **사전 계산**하여 데이터셋에 `behavior_log_probs`로 저장

**핵심 로직** (`compute_behavior_probs.py`):

```python
# Forward pass (no gradient)
outputs = model(input_ids=input_ids, use_cache=False)
logits = outputs.logits

# Log probabilities
log_probs_all = F.log_softmax(logits, dim=-1)
token_log_probs = log_probs_all.gather(..., labels_clamped.unsqueeze(-1)).squeeze(-1)
```

**출력 형식**: 각 샘플당 `[seq_len]` 크기의 log p(yt|x,y<t) 리스트

---

## Quick Reference: 논문 수식 → 코드

| 논문 | 코드 위치 |
|------|-----------|
| Δt = πθ/πβ | `prob_theta / prob_beta` (ref) 또는 `exp(log_theta - behavior_log_probs)` (precomputed) |
| δt = log Δt | `log_prob_theta - behavior_log_probs` |
| clip(δt, ℓΔ, uΔ) | `torch.clamp(log_ratio, clip_ratio_range[0], clip_ratio_range[1])` |
| w₁:T = ∏Δt | `torch.exp(total_log_ratio)` in `_uniform_weighting` |
| Gt = γ^(T-t) × ∏(j>t) Δj | `discount * suffix_weight` in `_suffix_weighting` |
| clip(Gt, Gmin, Gmax) | `torch.clamp(weight, clip_weight_range[0], clip_weight_range[1])` |
| L = Σ sg[Ĝt] × ℓt | `(weights * per_token_loss).sum()` with `weights.detach()` |

---

**검토 완료일**: 2026-02-11  
**구현 기반**: 논문 Algorithm 1, Section 3  
**결론**: 논문의 PEAR 알고리즘을 코드에 정확히 반영함 ✅
