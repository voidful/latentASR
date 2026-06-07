# LatentASR Threshold Sweep

Generated UTC: 2026-05-30T17:03:33+00:00

`Theta=-2.0` is the no-halting control: value-head scores are bounded in [-1, 1], so no early halt is triggered.
`dWER = Base WER - LR WER`; positive values mean LR is better. WER/CER and rates are percentages.

| Dataset | Theta | N | Avg steps | Skip N=0 | Full N=4 | Base WER | LR WER | dWER | Base CER | LR CER | dCER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLEURS en-US | -2.0 | 647 | 4.00 | 0.000 | 100.000 | - | 4.838 | - | - | 2.251 | - |
| FLEURS en-US | -0.2 | 647 | 1.37 | 43.740 | 21.484 | - | 4.859 | - | - | 2.269 | - |
| FLEURS en-US | 0.0 | 647 | 1.24 | 46.986 | 18.238 | - | 4.859 | - | - | 2.269 | - |
| FLEURS en-US | 0.2 | 647 | 0.00 | 100.000 | 0.000 | - | 4.900 | - | - | 2.326 | - |
| FLEURS en-US | 0.5 | 647 | 0.00 | 100.000 | 0.000 | - | 4.900 | - | - | 2.326 | - |
| VoxPopuli en | -2.0 | 1842 | 4.00 | 0.000 | 100.000 | - | 9.024 | - | - | 5.921 | - |
| VoxPopuli en | -0.2 | 1842 | 0.84 | 54.886 | 8.306 | - | 8.968 | - | - | 5.849 | - |
| VoxPopuli en | 0.0 | 1842 | 0.71 | 60.206 | 6.189 | - | 8.975 | - | - | 5.858 | - |
| VoxPopuli en | 0.2 | 1842 | 0.00 | 100.000 | 0.000 | - | 9.038 | - | - | 5.900 | - |
| VoxPopuli en | 0.5 | 1842 | 0.00 | 100.000 | 0.000 | - | 9.038 | - | - | 5.900 | - |

## Step Distributions

| Dataset | Theta | N=0 | N=1 | N=2 | N=3 | N=4 |
|---|---:|---:|---:|---:|---:|---:|
| FLEURS en-US | -2.0 | 0 | 0 | 0 | 0 | 647 |
| FLEURS en-US | -0.2 | 283 | 150 | 44 | 31 | 139 |
| FLEURS en-US | 0.0 | 304 | 150 | 45 | 30 | 118 |
| FLEURS en-US | 0.2 | 647 | 0 | 0 | 0 | 0 |
| FLEURS en-US | 0.5 | 647 | 0 | 0 | 0 | 0 |
| VoxPopuli en | -2.0 | 0 | 0 | 0 | 0 | 1842 |
| VoxPopuli en | -0.2 | 1011 | 476 | 141 | 61 | 153 |
| VoxPopuli en | 0.0 | 1109 | 443 | 124 | 52 | 114 |
| VoxPopuli en | 0.2 | 1842 | 0 | 0 | 0 | 0 |
| VoxPopuli en | 0.5 | 1842 | 0 | 0 | 0 | 0 |
