# Fold-CP 12SN E2E Baseline

This baseline was collected on 2026-07-16 from clean commit
`45185aed52d6f5474e831681d2e53704d3f2538e`. It covers 44 single-GPU and
four-GPU Fold-CP configurations, with and without the supplied 12SN MSA.

- Hardware: 8 x NVIDIA A800 80 GB; CP4 uses a `2 x 2` context-parallel mesh.
- Model: `opendde_v1`; BF16; sample=1; diffusion steps=2; cycle=1; seed=101.
- Kernels: `trimul=torch`, `triatt=torch`; TF32 disabled; LayerNorm=torch.
- Deterministic runs set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Determinism is enabled except for N=3072, which runs with determinism disabled.
- All 18 comparable single-GPU/CP4 pairs pass bitwise output alignment.
- Single-GPU cases at N>=2000 are omitted because they exceed single-card capacity.
- Speed ratio is `single-GPU time / CP4 time`: above 1 is faster; below 1 is slower.
- Memory improvement compares single-GPU peak with the peak on one CP4 rank/GPU.
- N=2800+MSA uses the successful isolated exact retry; the first primary attempt OOM is retained as diagnostic evidence.

| N | MSA | Single wall (s) | CP4 wall (s) | Wall speed | Single forward (s) | CP4 forward (s) | Forward speed | Single peak (GiB) | CP4 peak/GPU (GiB) | Per-GPU memory improvement | Status | Bitwise |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :---: | :---: |
| 101 | yes | 109.251 | 122.876 | 0.89× | 1.307 | 8.294 | 0.16× | 3.50 | 3.46 | 1.01×; 1.2% saved | ok | pass |
| 101 | no | 109.927 | 121.408 | 0.91× | 1.330 | 8.687 | 0.15× | 3.50 | 3.46 | 1.01×; 1.2% saved | ok | pass |
| 200 | yes | 110.907 | 122.440 | 0.91× | 1.959 | 8.943 | 0.22× | 6.64 | 5.10 | 1.30×; 23.1% saved | ok | pass |
| 200 | no | 109.261 | 123.832 | 0.88× | 1.924 | 8.712 | 0.22× | 6.64 | 5.10 | 1.30×; 23.1% saved | ok | pass |
| 299 | yes | 114.164 | 129.060 | 0.88× | 4.034 | 10.426 | 0.39× | 14.20 | 6.62 | 2.15×; 53.4% saved | ok | pass |
| 299 | no | 113.482 | 124.922 | 0.91× | 4.033 | 11.485 | 0.35× | 14.20 | 6.62 | 2.15×; 53.4% saved | ok | pass |
| 401 | yes | 117.333 | 127.914 | 0.92× | 7.507 | 13.566 | 0.55× | 28.55 | 10.80 | 2.64×; 62.2% saved | ok | pass |
| 401 | no | 116.177 | 129.850 | 0.89× | 7.482 | 14.141 | 0.53× | 28.55 | 10.79 | 2.64×; 62.2% saved | ok | pass |
| 600 | yes | 131.384 | 138.820 | 0.95× | 19.486 | 25.068 | 0.78× | 43.22 | 11.12 | 3.89×; 74.3% saved | ok | pass |
| 600 | no | 129.590 | 136.118 | 0.95× | 19.343 | 23.447 | 0.82× | 43.22 | 11.12 | 3.89×; 74.3% saved | ok | pass |
| 799 | yes | 222.244 | 153.086 | 1.45× | 44.926 | 35.959 | 1.25× | 44.12 | 20.07 | 2.20×; 54.5% saved | ok | pass |
| 799 | no | 220.414 | 153.926 | 1.43× | 44.814 | 39.378 | 1.14× | 44.11 | 20.07 | 2.20×; 54.5% saved | ok | pass |
| 1001 | yes | 256.687 | 170.883 | 1.50× | 82.835 | 57.406 | 1.44× | 66.93 | 35.41 | 1.89×; 47.1% saved | ok | pass |
| 1001 | no | 260.613 | 173.542 | 1.50× | 83.046 | 58.683 | 1.42× | 66.92 | 35.40 | 1.89×; 47.1% saved | ok | pass |
| 1200 | yes | 324.013 | 205.665 | 1.58× | 148.021 | 84.545 | 1.75× | 61.91 | 26.65 | 2.32×; 56.9% saved | ok | pass |
| 1200 | no | 326.248 | 200.863 | 1.62× | 148.110 | 83.070 | 1.78× | 61.90 | 26.64 | 2.32×; 57.0% saved | ok | pass |
| 1399 | yes | 427.399 | 242.561 | 1.76× | 248.408 | 126.411 | 1.97× | 64.17 | 34.76 | 1.85×; 45.8% saved | ok | pass |
| 1399 | no | 427.169 | 243.699 | 1.75× | 248.187 | 123.222 | 2.01× | 63.96 | 34.74 | 1.84×; 45.7% saved | ok | pass |
| 2000 | yes | — | 405.277 | N/A | — | 279.347 | N/A | — | 66.51 | N/A | ok | n/a |
| 2000 | no | — | 408.172 | N/A | — | 282.831 | N/A | — | 66.49 | N/A | ok | n/a |
| 2400 | yes | — | 657.281 | N/A | — | 529.335 | N/A | — | 61.79 | N/A | ok | n/a |
| 2400 | no | — | 659.775 | N/A | — | 528.303 | N/A | — | 61.77 | N/A | ok | n/a |
| 2800 | yes | — | 742.403 | N/A | — | 611.226 | N/A | — | 69.98 | N/A | ok | n/a |
| 2800 | no | — | 772.544 | N/A | — | 636.284 | N/A | — | 69.95 | N/A | ok | n/a |
| 3072 | yes | — | 815.566 | N/A | — | 669.825 | N/A | — | 65.72 | N/A | ok | n/a |
| 3072 | no | — | 820.474 | N/A | — | 683.988 | N/A | — | 65.69 | N/A | ok | n/a |

## N=2800+MSA diagnostic note

- The first primary attempt OOMed; an isolated exact retry with the same input,
  code, model, deterministic setting, kernels, image, and physical GPU group
  succeeded, and that retry is the row reported above.
