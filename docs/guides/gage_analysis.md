# Gage Study and measurement uncertainties

## Standard uncertainty components

The following table shows the typical standard uncertainties as they affect measurements. They are divided into Measurement System (MS) and Measurement Process (MP). The individual standard uncertainty components can be determined using method A or B. The method B refers to prior knowledge or information from manuals or specifications and the method A refers to statistical analysis.

| Shortcut | Uncertainty Component                                  | Impact | Method |
| :------: | :----------------------------------------------------- | :----: | :----: |
|   `RE`   | Display resolution                                     |   MS   |   B    |
|  `CAL`   | Calibration of the reference                           |   MS   |   B    |
|  `LIN`   | Linearity deviation                                    |   MS   |  A/B   |
|   `BI`   | Bias                                                   |   MS   |  A/B   |
|  `EVR`   | Measurement evolution (repeatability) on the reference |   MS   |   A    |
|  `MPE`   | Maximum Permissible Error (error limit)                |   MS   |   B    |
|  `EVO`   | Measurement evolution (repeatability) on the object    |   MP   |   A    |
|   `AV`   | Appraiser Variation (comparability of operators)       |   MP   |   A    |
|   `GV`   | Gage Variation (comparability of measurement system)   |   MP   |   A    |
|   `IA`   | Interactions                                           |   MP   |   A    |
|   `T`    | Temperature                                            |   MP   |  A/B   |
|  `STAB`  | Stability over time                                    |   MP   |   A    |
|  `OBJ`   | Inhomogeneity of the object                            |   MP   |  A/B   |
