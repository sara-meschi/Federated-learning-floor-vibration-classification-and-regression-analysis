| config | stage | E | optimizer state | steps | final std. MSE | distinct residuals | skill | detector | eligible | rank |
|---|---|---|---|---|---|---|---|---|---|---|
| fedavg_e01_fresh | stage_1 | 1 | fresh | 1800 | 1.0001 | 1 | +0.0036 | failed_detector | no | — |
| fedavg_e01_persistent | stage_1 | 1 | persistent | 1800 | 1.0000 | 238 | +0.0002 | failed_detector | no | — |
| fedavg_e05_fresh | stage_1 | 5 | fresh | 9000 | 1.0000 | 1 | +0.0011 | failed_detector | no | — |
| fedavg_e05_persistent | stage_1 | 5 | persistent | 9000 | 0.0549 | 238 | +0.1014 | passed | yes | 1 |
| fedavg_e10_fresh | stage_1 | 10 | fresh | 18000 | 0.0176 | 238 | -0.0105 | failed_detector | no | — |
| fedavg_e10_persistent | stage_1 | 10 | persistent | 18000 | 0.0220 | 238 | +0.3021 | passed | no | — |
| fedadam_e05_persistent | stage_2 | 5 | persistent | 9000 | 0.3808 | 238 | +0.0893 | passed | yes | 2 |
