# 아이디어 #2: MM-DiT 비대칭 가속화 (Asymmetric MM-DiT Acceleration)

## 핵심 통찰과 동기

MM-DiT (FLUX, SD3, SD3.5, Qwen-Image 등)는 text token과 image token을 **하나의 sequence로 합쳐서** joint attention을 수행하는 dual-stream 구조입니다. 그런데 두 종류의 token은 본질적으로 매우 다른 특성을 가집니다:

- **Text token**: 보통 256~512개 정도로 적음. Denoising step이 진행되어도 **거의 변화하지 않음** (text embedding은 거의 정적). 그런데 transformer block마다 매번 재계산됨.
- **Image token**: FLUX 1024×1024에서 4096개, 고해상도면 16K+. **Step별로 빠르게 진화**하지만, 인접 timestep끼리는 유사도가 높음.
- **Joint attention**: $O((N_{text} + N_{image})^2)$인데 $N_{image} \gg N_{text}$이므로 사실상 image-image, image-text, text-image, text-text 네 블록 중 image-image가 압도적 비중.

기존 caching/sparse attention 연구들은 모두 두 stream을 **uniform하게** 처리합니다. ToCa, DuCa, FORA 등 token-level caching도 text/image 구분 없이 token importance만 봅니다. 이것이 큰 빈 자리입니다.

## 해결할 구체적 문제

1. Text token에 대한 **불필요한 재계산** (step별로 거의 안 변하는데 매번 forward).
2. Cross-modal attention (image↔text)의 redundancy를 활용하지 못함.
3. Image token만 aggressive cache하면 cross-attention path에서 stale text representation과 mismatch 발생.

## 구체적 방법론 설계

### (a) Modality-Asymmetric Caching Schedule
- **Text stream**: 초기 5~10 step에서만 계산하고 이후로는 freeze (또는 ~10 step마다만 refresh)
- **Image stream**: 기존 token-level caching (예: 2 step 간격)
- **Cross-attention KV**: text-side KV는 거의 정적이므로 한 번 계산 후 long horizon으로 reuse. 이게 큰 win.

### (b) Stream-Specific Block Skipping
DiT block 내부에서 text branch와 image branch가 분리되어 modulation/MLP를 따로 가집니다. 각 branch의 contribution을 측정해서:
- Text branch의 self-modulation은 후반 step에서 거의 영향 없음 → skip
- Image branch는 후반 step에서 detail에 critical → keep

### (c) Asymmetric Sparse Attention Mask
Joint attention matrix를 4개 블록으로 분해:

$$
A = \begin{bmatrix} A_{TT} & A_{TI} \\ A_{IT} & A_{II} \end{bmatrix}
$$

- $A_{TT}$ (text-text): 매우 작음, dense 유지
- $A_{II}$ (image-image): 가장 큼, aggressive sparse (block sparse, 60-80% sparsity)
- $A_{TI}, A_{IT}$ (cross): mid sparsity, head-adaptive

FLUX의 single-stream block과 double-stream block에서 각각 다른 전략 적용 가능.

### (d) FLUX 특화: Single/Double-Stream 차별화
FLUX는 처음 19개가 double-stream block, 뒤 38개가 single-stream block입니다. Single-stream에서는 이미 두 modality가 합쳐졌으므로 caching이 다르게 작동해야 함. Layer-depth별 strategy를 별도로 학습.

## 구체적 실험 디자인

**Step 1: 분석 실험 (필수, 동기 부여용)**
- FLUX/SD3에서 각 step마다 text token feature의 L2 변화량과 image token feature의 L2 변화량을 측정 → 큰 차이 입증
- Cross-attention attention map의 sparsity 분석 → text-image attention이 더 sparse한지 확인
- Block별 ablation: text branch만 skip vs image branch만 skip의 quality 영향

**Step 2: Training-free 버전**
- FLUX.1-dev (28 step), SD3 (28 step), Qwen-Image
- Baselines: FORA, ToCa, DuCa, TaylorSeer, SmoothCache
- Metrics: FID, CLIPScore, ImageReward, PickScore, wall-clock latency (H100)
- 평가 dataset: COCO 30K, MJHQ-30K, GenEval (prompt 정합성)

**Step 3: Fine-tuning 버전 (optional)**
- Asymmetric schedule을 explicit하게 학습하기 위한 LoRA-based fine-tuning (1-2일이면 충분)
- 더 aggressive한 acceleration ratio 달성

**Step 4: 일반화 실험**
- Video MM-DiT (HunyuanVideo, Wan2.1)에도 적용
- 다른 modality (e.g., Stable Audio Open)에도 transfer 가능성

## 기대 효과 (구체적 수치)

기존 baseline 대비:
- **단순 caching (FORA/ToCa)**: FLUX에서 1.5~2.0× speedup, FID 약간 손실
- **DuCa/TaylorSeer**: FLUX 2.5~3.0× speedup
- **본 방법 예측**: **3.5~5.0× speedup**, near-lossless quality (FID 변화 <0.5, CLIPScore 변화 <0.01)
  - Text stream 거의 freeze로 텍스트 path 90% 절약
  - Cross-attention KV reuse로 추가 20-30% 절약
  - 이미지 caching은 기존 SOTA 수준 유지

H100에서 FLUX.1-dev 1024×1024 28 step:
- Baseline: ~4-5초
- 본 방법: **1.0~1.3초 수준 예상**

## 위험 요소와 대응

- **위험 1**: Text stream을 너무 일찍 freeze하면 prompt alignment 손상 (GenEval 점수 하락)
  - **대응**: 분석 실험으로 safe freeze timing 찾기. Prompt complexity별로 schedule 차별화.
- **위험 2**: FLUX의 RoPE가 sparse attention과 안 맞음 (SDTM 논문에서 언급)
  - **대응**: RoPE에 친화적인 block-aligned sparsity pattern 설계.
- **위험 3**: 이미 누가 비슷한 아이디어를 진행 중일 수 있음
  - **대응**: 단순 "asymmetric"이 아니라 **이론적 분석 (modality별 information bottleneck 분석)** 을 추가해 차별화.

## 논문 스토리

"우리는 MM-DiT에서 text와 image stream의 dynamics가 본질적으로 비대칭임을 보이고, 이 비대칭성을 활용해 cross-modal aware한 통합 가속화 framework를 제안한다." — CVPR/ICCV/NeurIPS 모두 fit.

## 요약 정보

| 항목 | 내용 |
|---|---|
| **타겟 모델** | FLUX.1-dev, SD3, SD3.5, Qwen-Image, HunyuanVideo |
| **Novelty 강도** | 中-高 (구조적 통찰) |
| **이론적 깊이** | 中 |
| **구현 난이도** | 中 (training-free 가능) |
| **컴퓨팅 자원 요구** | 中 (FLUX inference만) |
| **기대 speedup** | 3.5-5× |
| **타겟 학회** | CVPR / ICCV |
| **예상 시간 소요** | 3-4개월 |

---

## Step 1 분석 실험 결과 (2026-05-14)

**결론: 핵심 가설이 FLUX.1-dev와 SD3-medium 모두에서 성립하지 않음. 아이디어 재검토 필요.**

### 실험 A: Stream Dynamics (text vs image 변화량)

**가설**: step이 진행될수록 text token이 거의 변하지 않고 image token이 크게 변함 (text_delta / image_delta < 0.30).

| 모델 | text_mean_delta | image_mean_delta | ratio | 판단 |
|---|---|---|---|---|
| FLUX.1-dev | 0.0716 | 0.0306 | **2.34** | ❌ FAIL |
| SD3-medium | 0.1838 | 0.0893 | **2.06** | ❌ FAIL |

- **반대 현상 관찰**: text token이 image token보다 2배 이상 빠르게 변함
- text stream은 denoising 과정에서 context 정보를 적극적으로 재구성하며, image stream은 상대적으로 안정적
- 이 결과는 아이디어의 **핵심 전제를 파괴**함 (plots: `results/plots/exp_a_flux_full_analysis.png`, `exp_a_sd3_full_analysis.png`, `exp_a_comparison.png`)

### 실험 B: Attention Block Sparsity (FLUX.1-dev)

**가설**: $A_{II}$ (image-image) entropy < 50% of $A_{TI}$ (text-image) entropy → asymmetric sparse mask 정당화.

| 항목 | 값 |
|---|---|
| II entropy (mean) | 5.67 |
| TI entropy (mean) | 6.49 |
| II / TI ratio | **0.87** |
| 판단 | ⚠️ REVIEW (need < 0.50) |

- 4개 quadrant 모두 entropy가 비슷 (5.5~6.5 nats). A_II가 약간 낮지만 기준치(0.50)에 한참 못 미침
- Asymmetric sparse mask 설계 정당화 어려움 (plots: `results/plots/exp_b_flux_entropy_heatmap.png`, `exp_b_flux_entropy_bar.png`)

**SD3-medium**: II entropy=3.64, TI entropy=3.49, ratio=1.045 → ⚠️ REVIEW
- FLUX와 **반대 패턴**: A_TI entropy(3.49) < A_II entropy(3.64). 즉 cross-modal attention이 image-image보다 오히려 더 집중됨
- 두 모델 모두 asymmetric sparse mask를 정당화하지 못하나, 실패 방향이 정반대 (FLUX: A_II가 약간 낮음, SD3: A_TI가 약간 낮음)
- *(참고: `logs/full_b_sd3.log`에 verdict=PASS/ratio=0.0 출력은 로깅 버그. 실제값은 `results/attn_sparsity_sd3.json` 기준)*

### 실험 D: Cross-Attention KV Drift

**가설**: text-side K/V는 step에 따라 거의 변하지 않아 reuse 가능 (final step drift < 10%).

| 모델 | K_drift (step 27) | V_drift (step 27) | 판단 |
|---|---|---|---|
| FLUX.1-dev | **67.0%** | **90.5%** | ❌ FAIL *(코드 verdict: REVIEW — 기준 <10% 미달)* |
| SD3-medium | **105.6%** | **113.6%** | ❌ FAIL *(코드 verdict: REVIEW — 100%+ drift는 실질적 반전)* |

- text-side KV는 step 0에서 최종 step까지 위치를 완전히 바꿈 (100%+ drift는 반대 방향까지 이동)
- KV reuse는 완전히 불가 (plots: `results/plots/exp_d_flux.png`, `exp_d_sd3.png`)

**SD3 KV freeze 품질 측정** (`results/freeze_text_kv_quality.csv`):

| freeze 시작 step | PSNR |
|---|---|
| no_freeze | 62.99 dB |
| step 1부터 freeze | 10.43 dB (catastrophic) |
| step 5부터 freeze | 16.30 dB (catastrophic) |

- freeze 시작을 step 5로 늦춰도 품질 붕괴는 동일하게 발생 → KV reuse 여지 없음

### 실험 C: Branch Ablation (완료 — 2026-05-14)

**가설**: text branch skip이 image branch skip보다 품질 영향이 작음 (text_SSIM_drop / image_SSIM_drop < 0.30).

**측정**: step 15–27에서 text branch 또는 image branch를 완전히 skip한 뒤 SSIM, PSNR, CLIPScore 비교.

| 모델 | schedule | SSIM | PSNR | CLIP | SSIM drop |
|---|---|---|---|---|---|
| FLUX.1-dev | baseline | 0.9999 | 92.90 | 30.14 | — |
| FLUX.1-dev | text_skip_late | 0.8924 | 27.14 | 29.47 | 0.1075 |
| FLUX.1-dev | image_skip_late | 0.1819 | 12.46 | 23.29 | 0.8180 |
| SD3-medium (n=200)¹ | baseline | 0.9999 | 89.40 | 30.16 | — |
| SD3-medium | text_skip_late | 0.0841 | 7.37 | 14.89 | 0.9158 |
| SD3-medium | image_skip_late | 0.1779 | 9.59 | 19.34 | 0.8220 |

| 모델 | ratio (t_drop/i_drop) | 판단 |
|---|---|---|
| FLUX.1-dev | **0.1315** | ✅ PASS |
| SD3-medium | **1.114** | ❌ FAIL |

**해석**:
- **FLUX**: text branch skip이 image branch skip 대비 피해 13% → text branch는 late step에서 상대적으로 덜 중요. FLUX의 38개 single-stream block에서 text/image가 이미 fused되어 double-stream text branch 기여가 희석되기 때문으로 추정.
- **SD3**: text branch skip이 image branch skip보다 오히려 더 큰 피해 (111%) → pure double-stream 구조에서 text branch가 late step까지 필수. 아이디어 전제 완전 불성립.
- (plots: `results/plots/branch_ablation_flux.png`, `branch_ablation_sd3.png`)

> ¹ SD3 n=200: GPU 2대(각 100개) 병렬 실행 후 합산 (`branch_ablation_sd3_g0.csv` + `branch_ablation_sd3_g3.csv`). FID는 Exp C에서 미측정 (reference set 미구성).

---

### 종합 분석: 왜 아이디어가 작동하지 않는가

```
[원래 가정]                         [실제 관찰]
text tokens: 정적 (frozen)    →    text tokens: 동적 (2× more than image!)
image tokens: 동적            →    image tokens: 상대적으로 안정적
text KV: 재사용 가능          →    text KV: 100%+ drift, 재사용 불가
A_II: sparse                 →    A_II ≈ A_TI ≈ A_TT (모두 비슷)
text branch skip-safe (late) →    FLUX: 일부 OK (ratio=0.13), SD3: 완전 불가 (ratio=1.11)
```

**Step 1 실험 전체 결과표**:

| 실험 | 모델 | 결과 | 판단 |
|---|---|---|---|
| A (dynamics) | FLUX | text/image ratio=2.34 | ❌ FAIL |
| A (dynamics) | SD3 | ratio=2.06 | ❌ FAIL |
| B (sparsity) | FLUX | II/TI entropy=0.87 | ⚠️ REVIEW |
| B (sparsity) | SD3 | II/TI entropy=1.05 | ⚠️ REVIEW |
| C (ablation) | FLUX | ratio=0.1315 | ✅ PASS |
| C (ablation) | SD3 | ratio=1.114 | ❌ FAIL |
| D (KV drift) | FLUX | K=67%, V=91% | ❌ FAIL |
| D (KV drift) | SD3 | K=106%, V=114% | ❌ FAIL |
| D (freeze) | SD3 | PSNR: 63→10 dB | ❌ FAIL |

**핵심 이유**: MM-DiT의 text stream은 단순한 "조건 embedding"이 아니라 **denoising 과정에서 image feature와 상호작용하면서 적극적으로 변화함**. Text token은 image token으로부터 cross-attention으로 영향을 받아 매 step 업데이트됨. 이는 text feature가 "정적 conditioning"이 아닌 "dynamic joint processing"의 참여자임을 의미.

**예외적 발견 (FLUX Exp C PASS)**: FLUX에서만 text branch skip이 유의미하게 덜 파괴적 (ratio=0.13). 이는 FLUX의 하이브리드 구조(double+single stream) 특성으로, single-stream block 38개에서 이미 text-image fusion이 일어나기 때문에 double-stream text branch의 후반 기여가 상대적으로 작아지는 현상. 단, 이것만으로는 "text가 정적"이라는 원래 전제를 지지하지 못함 (Exp A 참조).

**결론**: 아이디어 2의 핵심 전제 (text stream 정적, KV reuse 가능)가 두 모델 모두에서 성립하지 않음. 현재 형태로는 Step 2 진행 불가. 단, FLUX의 Exp C 결과는 아래 Option 2 방향의 단서가 됨.

---

### 다음 방향 (Option)

1. **Pivot: Cross-modal redundancy 측면** — text-image 간 cross-attention에서 redundant head/layer를 찾아 skip하는 방향 (dynamics asymmetry 없이도 가능)
2. **Pivot: FLUX double-stream text branch lazy-update** — Exp C에서 FLUX text branch가 덜 민감함을 확인. Double-stream block의 text branch만 2~3 step 간격으로 refresh하는 lightweight caching 탐색. SD3에서는 미적용.
3. **Pivot: Image-side late-step stability** — image token의 상대적 안정성(ratio 역수 ≈ 0.43)을 활용한 image branch caching (Exp A 결과의 역방향 활용)
4. **다른 아이디어로 이동** — Step 1 결과를 바탕으로 다른 가속화 아이디어 탐색

---

### 최종 결정 (2026-05-14)

**아이디어 폐기.** Step 1 핵심 가설 4개 중 3개 완전 실패 (Exp A, D: 두 모델 모두 FAIL, Exp C: SD3 FAIL), 1개 부분 통과 (Exp C FLUX only). 원래 전제인 "text stream 정적 + KV reuse 가능"이 두 모델 모두에서 성립하지 않아 현재 형태로는 Step 2 진행 불가.

Option 2 (FLUX text branch lazy-update)와 Option 3 (image-side caching)은 독립적인 아이디어로 분리해 별도 검토 가능하나, 본 workspace에서는 추가 진행하지 않음.
