# Workspace_mmdit_asymmetric

**Asymmetric MM-DiT Acceleration** — Step 1 분석 실험

MM-DiT(FLUX, SD3 등)에서 text/image stream의 dynamics 비대칭성을 측정하고,
이를 기반으로 Modality-Asymmetric Caching을 설계하는 연구의 실험 워크스페이스.

아이디어 전문: [`idea2_mmdit_asymmetric.md`](idea2_mmdit_asymmetric.md)

---

## 구조

```
configs/          # 모델별 생성/평가 설정 (FLUX, SD3)
analysis/         # Step 1 분석 스크립트
  hook_utils.py             # forward hook 유틸리티
  measure_stream_dynamics.py   # 실험 A: text/image L2 dynamics
  measure_attention_sparsity.py # 실험 B: 4-quadrant attention sparsity
  measure_branch_ablation.py    # 실험 C: text-branch vs image-branch skip
  measure_xattn_kv_drift.py    # 실험 D: cross-attn text-side KV drift
  plot_results.py              # heatmap / bar / line plot 생성
eval/
  eval_utils.py   # FID, CLIPScore, ImageReward 통합
  prompts/        # prompt 파일
  reference/      # 레퍼런스(no-caching) 이미지
baselines/        # Step 2 준비 — FORA, SmoothCache, ToCa/DuCa, TaylorSeer
results/          # JSON/CSV 로그
  plots/          # 생성 플롯
scripts/          # 실행 스크립트
logs/
```

---

## 빠른 시작 (Quick Start)

```bash
# 1. 레퍼런스 이미지 생성 (smoke test: 10 prompts)
bash scripts/run_reference_gen.sh --quick

# 2. 분석 실험 A–D 실행 (smoke test)
bash scripts/run_step1_analysis.sh --quick

# 3. 플롯 생성
python analysis/plot_results.py --all
```

---

## 실험 요약

| 실험 | 측정 대상 | 판단 기준 | 스크립트 |
|---|---|---|---|
| **A** | text vs image stream 변화율 | text/image ratio < 0.30 | `measure_stream_dynamics.py` |
| **B** | A_II vs A_TI entropy | II/TI < 0.50 | `measure_attention_sparsity.py` |
| **C** | text-branch skip quality 영향 | text skip FID drop < image skip × 0.30 | `measure_branch_ablation.py` |
| **D** | text-side KV step별 drift | 최종 step drift < 10% | `measure_xattn_kv_drift.py` |

---

## 타겟 모델

| 모델 | config |
|---|---|
| FLUX.1-dev (28 step, 1024×1024) | `configs/flux_dev.yaml` |
| SD3-medium (28 step, 1024×1024) | `configs/sd3_medium.yaml` |

하드웨어: NVIDIA B200 (183GB VRAM)
