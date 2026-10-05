# Results

## Sample efficiency

130 runs, 2,600,000 held-out shots. Error per cycle is fitted over rounds per patch and basis, then averaged. paired z > 0 means more logical errors than the vendor prior on the same shots.

### Distance 3

| prior | shots used | logical errors | error per cycle % | vs vendor % | paired z |
|---|---|---|---|---|---|
| Google tensor network (stronger decoder) | 0 | 511,066 | 3.072 | -23.7 | -118.9 |
| Google belief matching (stronger decoder) | 0 | 522,034 | 3.148 | -21.8 | -116.1 |
| Google correlated matching (stronger decoder) | 0 | 554,275 | 3.500 | -13.0 | -67.7 |
| Google's own PyMatching run | 0 | 593,313 | 4.024 | -0.0 | -0.8 |
| Vendor prior (Google benchmarks) | 0 | 593,326 | 4.025 | +0.0 | +0.0 |
| Uniform prior (no knowledge) | 0 | 622,886 | 4.437 | +10.2 | +45.7 |
| shrinkdem v2 around vendor prior | 250 | 611,664 | 3.903 | -3.0 | +33.1 |
| shrinkdem v1 (pooling) | 250 | 609,825 | 3.925 | -2.5 | +29.0 |
| shrinkdem v2 (pooling + budgets) | 250 | 609,825 | 3.925 | -2.5 | +29.0 |
| Raw p_ij, floor 0.001 | 250 | 678,399 | 5.483 | +36.2 | +113.9 |
| Raw p_ij | 250 | 689,872 | 5.613 | +39.5 | +127.2 |
| shrinkdem v1 (pooling) | 1,000 | 593,221 | 3.916 | -2.7 | -0.2 |
| shrinkdem v2 (pooling + budgets) | 1,000 | 593,221 | 3.916 | -2.7 | -0.2 |
| shrinkdem v2 around vendor prior | 1,000 | 592,222 | 3.936 | -2.2 | -2.1 |
| Raw p_ij, floor 0.001 | 1,000 | 622,163 | 4.637 | +15.2 | +45.0 |
| Raw p_ij | 1,000 | 625,896 | 4.683 | +16.4 | +50.4 |
| shrinkdem v2 around vendor prior | 4,000 | 586,223 | 3.935 | -2.2 | -13.9 |
| shrinkdem v1 (pooling) | 4,000 | 587,349 | 4.001 | -0.6 | -11.7 |
| shrinkdem v2 (pooling + budgets) | 4,000 | 587,349 | 4.001 | -0.6 | -11.7 |
| Raw p_ij, floor 0.001 | 4,000 | 597,073 | 4.171 | +3.6 | +6.7 |
| Raw p_ij | 4,000 | 598,698 | 4.197 | +4.3 | +9.5 |
| shrinkdem v2 around vendor prior | 16,000 | 584,818 | 3.956 | -1.7 | -16.5 |
| shrinkdem v1 (pooling) | 16,000 | 584,889 | 3.966 | -1.5 | -16.3 |
| shrinkdem v2 (pooling + budgets) | 16,000 | 584,889 | 3.966 | -1.5 | -16.3 |
| Raw p_ij, floor 0.001 | 16,000 | 588,921 | 4.073 | +1.2 | -8.3 |
| Raw p_ij | 16,000 | 589,774 | 4.095 | +1.8 | -6.7 |
| Google p_ij model, 25,000 shots | 25,000 | 583,412 | 3.909 | -2.9 | -21.7 |
| shrinkdem v1 (pooling) | 30,000 | 585,166 | 3.969 | -1.4 | -15.8 |
| shrinkdem v2 (pooling + budgets) | 30,000 | 585,166 | 3.969 | -1.4 | -15.8 |
| shrinkdem v2 around vendor prior | 30,000 | 584,722 | 3.971 | -1.3 | -16.7 |
| Raw p_ij, floor 0.001 | 30,000 | 587,310 | 4.058 | +0.8 | -11.5 |
| Raw p_ij | 30,000 | 588,502 | 4.076 | +1.3 | -9.2 |

### Distance 5

| prior | shots used | logical errors | error per cycle % | vs vendor % | paired z |
|---|---|---|---|---|---|
| Google tensor network (stronger decoder) | 0 | 119,685 | 2.983 | -32.7 | -82.7 |
| Google belief matching (stronger decoder) | 0 | 123,808 | 3.128 | -29.5 | -75.8 |
| Google correlated matching (stronger decoder) | 0 | 136,120 | 3.641 | -17.9 | -41.4 |
| Vendor prior (Google benchmarks) | 0 | 149,870 | 4.435 | +0.0 | +0.0 |
| Google's own PyMatching run | 0 | 149,997 | 4.466 | +0.7 | +2.3 |
| Uniform prior (no knowledge) | 0 | 169,541 | 5.795 | +30.7 | +51.2 |
| shrinkdem v2 around vendor prior | 250 | 152,183 | 4.485 | +1.1 | +8.9 |
| shrinkdem v2 (pooling + budgets) | 250 | 153,466 | 4.505 | +1.6 | +12.8 |
| shrinkdem v1 (pooling) | 250 | 155,000 | 4.576 | +3.2 | +17.0 |
| Raw p_ij, floor 0.001 | 250 | 180,973 | 6.938 | +56.4 | +76.2 |
| Raw p_ij | 250 | 183,323 | 7.136 | +60.9 | +81.1 |
| shrinkdem v2 (pooling + budgets) | 1,000 | 150,453 | 4.380 | -1.2 | +2.3 |
| shrinkdem v2 around vendor prior | 1,000 | 150,348 | 4.407 | -0.6 | +2.0 |
| shrinkdem v1 (pooling) | 1,000 | 152,027 | 4.577 | +3.2 | +7.4 |
| Raw p_ij, floor 0.001 | 1,000 | 163,809 | 5.630 | +27.0 | +38.5 |
| Raw p_ij | 1,000 | 164,581 | 5.682 | +28.1 | +40.4 |
| shrinkdem v2 around vendor prior | 4,000 | 149,595 | 4.403 | -0.7 | -1.2 |
| shrinkdem v2 (pooling + budgets) | 4,000 | 149,631 | 4.407 | -0.6 | -1.0 |
| shrinkdem v1 (pooling) | 4,000 | 151,162 | 4.690 | +5.7 | +4.4 |
| Raw p_ij, floor 0.001 | 4,000 | 155,573 | 5.118 | +15.4 | +17.6 |
| Raw p_ij | 4,000 | 155,937 | 5.164 | +16.4 | +18.6 |
| shrinkdem v2 around vendor prior | 16,000 | 149,031 | 4.392 | -1.0 | -3.7 |
| shrinkdem v2 (pooling + budgets) | 16,000 | 149,087 | 4.397 | -0.8 | -3.4 |
| shrinkdem v1 (pooling) | 16,000 | 151,440 | 4.753 | +7.2 | +5.3 |
| Raw p_ij, floor 0.001 | 16,000 | 152,982 | 4.882 | +10.1 | +10.1 |
| Raw p_ij | 16,000 | 153,300 | 4.919 | +10.9 | +11.0 |
| Google p_ij model, 25,000 shots | 25,000 | 148,972 | 4.428 | -0.2 | -3.7 |
| shrinkdem v2 (pooling + budgets) | 30,000 | 148,952 | 4.386 | -1.1 | -3.9 |
| shrinkdem v2 around vendor prior | 30,000 | 148,976 | 4.389 | -1.0 | -3.9 |
| Raw p_ij, floor 0.001 | 30,000 | 152,419 | 4.813 | +8.5 | +8.4 |
| shrinkdem v1 (pooling) | 30,000 | 151,703 | 4.814 | +8.5 | +6.2 |
| Raw p_ij | 30,000 | 152,824 | 4.869 | +9.8 | +9.6 |

## Stale prior versus fresh prior

Logical errors when each run is replayed in order (blocks of 2,000 shots after the first 4,000).

| distance | runs | n runs | shots | vendor | static 4k | trailing 4k | expanding | trailing vs vendor % | expanding vs vendor % |
|---|---|---|---|---|---|---|---|---|---|
| 3 | all runs | 24 | 1,104,000 | 465,908 | 463,433 | 462,751 | 462,385 | -0.68 | -0.76 |
| 3 | drifting runs (|z| >= 5) | 6 | 276,000 | 118,010 | 118,783 | 118,096 | 118,090 | +0.07 | +0.07 |
| 3 | stable runs | 18 | 828,000 | 347,898 | 344,650 | 344,655 | 344,295 | -0.93 | -1.04 |
| 5 | all runs | 22 | 1,012,000 | 337,619 | 336,639 | 336,276 | 335,645 | -0.40 | -0.58 |
| 5 | drifting runs (|z| >= 5) | 5 | 230,000 | 71,759 | 71,905 | 71,678 | 71,582 | -0.11 | -0.25 |
| 5 | stable runs | 17 | 782,000 | 265,860 | 264,734 | 264,598 | 264,063 | -0.47 | -0.68 |

## Burst monitor (repetition code, 500,000 shots)

Flagged 520 shots in 4 bursts. They contain 31 of the 35 logical errors of the whole run. Logical error per shot: 7.0e-05 with every shot, 8.0e-06 without the flagged ones.

| first shot | last shot | peak z | logical errors inside |
|---|---|---|---|
| 38460 | 38519 | 10.4 | 0 |
| 57740 | 57999 | 57.8 | 31 |
| 177640 | 177759 | 15.9 | 0 |
| 451380 | 451459 | 12.6 | 0 |
