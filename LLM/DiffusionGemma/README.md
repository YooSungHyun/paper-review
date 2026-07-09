# DiffusionGemma까지 달려보기

## 논문

1. https://arxiv.org/abs/2406.07524
2. https://arxiv.org/abs/2406.04329
3. https://arxiv.org/abs/2502.09992
4. https://arxiv.org/abs/2508.15487
5. https://arxiv.org/abs/2503.09573

## 요약

### [1. Masked Diffusion LM (24.06)](https://arxiv.org/abs/2406.07524)

![image-20260701194003829](./img/1.png)

- Diffusion은 이미지에 주로 쓰다보니 가우시안 노이즈를 주기 쉬웠음 (픽셀은 연속값이니까.). 근데 LM에는 주기 힘듬
- 따라서, 2가지 방법이 존재했음. 하나는 Embedding에 noise주기, 하나는 Token에 Noise주기. Token을 단 랜덤 토큰으로 치환하던가 하다보니, Entropy(PPL)이 높아지는 문제가 발생함. (모델이 잘 못함)
- Diffusion은 크게 Loss가 3개인데(Recon + Diffusion + Prior), Noise를 [MASK]로 줬더니, Recon, Prior날라가고 Diffusion loss마저도 일반 CE Loss로 일반화됨.
- 즉, 앞으로 Diffusion LM은 [MASK]로 랜덤하게 주고 BERT맹키로 Masked LM으로 학습시킨다음에, 첫 시작때 전부 MASK주고 뺑뺑이 돌리다보면, 예측이 잘 될것!
  - **여기서 문제는 이미 예측된 문자열을 바꿀 수가 없으니**, 모델이 멍청하게 시작하면 문장이 반드시 고장남 -> 이후 DiffusionGemma에서 해당 문제를 개선함.
- 또한, Inference에서 긴 문장을 예측할때 예를 들어, 4개씩 계속 mask를 붙혀서 예측시키면, Multi Token Prediction처럼 Semi AR 처럼 예측 가능할 것! -> 이후에 Block Wise Inference에서 다룸.

### [2. Simplified and Generalized Masked Diffusion for Discrete Data (24.06)](https://arxiv.org/abs/2406.04329)

- 비슷한 시기에 나온 논문이지만 Google DeepMind에서 썼고, 대부분의 핵심골자는 거의 완전히 유사하다.
- 다만, 이쪽이 조금 더 수학적으로 LLM의 Diffusion이 왜 CE Loss에 근사하는지 더 구체적으로 작성되어있다. (Diffusion Gemma를 이해하기위해 이걸 다 증명하고있을 필요는 없다고 생각됨.)
- 중요한건 MDLM은 **Inference Sampling이 하이퍼파라미터로, 사람이 입력하지만, MD4는 이마저도 Scheduler로 고민을 했다는 큰 차이점**이 있다.
  - 한번에 많은 양을 선형적으로 증가하며 예측시키면 Conflict가 나니까, Cosine같이 점차적으로 문장이 완성되어감에 따라, 더 많은 토큰을 예측할 수 있도록 하는게 좋았다는 얘기
  - 그리고 time step역시 고민을 하는데, 그냥 균등하게 자르는게 제일 좋았다고함.
- MDLM은 모든 토큰이 균일한 가중치로 오픈되는데, **어떤 토큰은 우선순위가 더 높을 수도 있고, 이걸 학습으로 해결할 수 있다.**(다만 Gradient Graph가 깨지는 방법론이라, 굉장히 구현적으로 어려움....;;, 왜냐면 vocab에 해당하는 learnable w 1차원 행렬로 베르누이 분포로 뽑는데, 결국 w는 float라서, 이녀석이 0.0001올랐을때 샘플링된 결과는 어떻게 연속적으로 변할껀데? -> 이걸 계산할 수 없는 방식이라 미분 불가능함.)
- 이 논문은 **자연어 뿐만 아니라 모든걸 Discrete하게 해석할 수 있다고 주장**하며, 이미지를 Discrete하게 치환하여 했는데, 일반적인 이미지모델보다 성능이 훨씬 좋았다고함.
- 구현을 굉장히 유심히 해야함을 명시하는데, Categorical Sampling을 할때, Softmax를 사용하는데, JAX의 경우 Gumble Softmax를 사용하면서, 의도치 않은 Numerical Stablity에 의한 오차가 발생한다고함. (vLLM같은거 구현 잘못하면 모델 학습 잘해도 Inference에서 성능이 개 떨어져보일 수 있으니 조심하라고함.)

### [3. Large Language Diffusion Models (LLaDA) (25.02)](https://arxiv.org/abs/2502.09992)

- 세훈이가 리뷰했던 그 논문 (자세한건 노션 확인)
- 1,2번은 그냥 일반 Decoder에 대해 논하지만, 3번은 LLM식으로 SFT학습을 할 수 있다고 하는 것.
- 방식은 그냥 Input은 마스킹 안하고 Response에 대해서만 MDLM 공식을 적용하면 됨.
- 다만 특별한 점은, 여기는 Inference에서 토큰 N개를 예측하는 기준을, 일단 전체 토큰을 Inference한 다음, Token Prob이 낮은 하위 M개에 대해서 마스킹으로 두어, 다시 예측할 수 있도록 한다.
- 즉, 여기도 이미 뽑힌 토큰에 대해서 수정은 불가하며, 뽑히기 전에 단순히 확률이 상대적으로 낮은 토큰들은 패를 안까겠다는 전략.

### [4. Dream 7B: Diffusion Large Language Models (25.08)](https://arxiv.org/abs/2508.15487)

- LLaDA는 LLM을 Scratch부터 Pre Training해야하니 개빡세다!!! -> 그냥 AR모델 가져다가 Diffusion으로 마개조하면 안됨? -> 실제로 잘됨. (토큰 거의 4배 미만으로 씀)

![2](./img/2.png)

- 다만 이 방식은, AR로 강하게 학습되어있으니까, Diffusion Model은 t 타임에 t를 예측하지만, Diffusion Concept을 따르되, t타임에 t+1을 예측하도록 학습시킨다. (Shifted Prediction; 말이 어렵지 그냥 label shift임)
- loss 설계도 좀 다른데, 근처에 토큰이 많이 열려있는 경우와 그렇지 않은 경우 난이도 차이가 있으니까, Context-Adaptive noise Rescheduling at Token-level(**CART**)로 보정을 시켜준다. 각 token별 label prob에 가중치를 주는 형태로 한다.

### [5. Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (25.03)](https://arxiv.org/abs/2503.09573)

![3](./img/3.png)

- Block을 걸어서 Block단위 Auto-Regressive하게 Diffusion을 진행하면, 앞 Block까지는 KV caching이 사용 가능하니까 개꿀이다. 단, Block을 잘게 넣으면 속도가 오래걸리는 문제가 발생.
- 

## DiffusionGemma
