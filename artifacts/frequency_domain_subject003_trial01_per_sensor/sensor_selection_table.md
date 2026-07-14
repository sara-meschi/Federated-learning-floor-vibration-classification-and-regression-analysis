# Trial 1 Sensor Selection Table

Quiet baseline is the pre-stomp region 0.5 s to 9.295 s. Walking windows are detected separately for each sensor.

| sensor | location_y | walk_window_s | time_snr_db | walk_20_50_snr_db | line_115_125_snr_db | full_0p5_200_snr_db | peak_hz | peak_type | timing_vs_location | score | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 1105.0 | 16.23-18.23 | 0.99 | 1.14 | 0.19 | 0.51 | 120.16 | fixed-line/artifact? | poor | -5 | DROP |
| 12 | 1105.0 | 11.96-13.96 | 0.39 | -0.18 | -0.06 | -0.05 | 120.16 | fixed-line/artifact? | good | -3 | DROP |
| 13 | 1105.0 | 15.08-17.08 | 0.74 | 2.18 | -0.59 | 0.87 | 120.16 | fixed-line/artifact? | good | -3 | DROP |
| 14 | 1010.0 | 15.83-17.83 | 0.24 | 3.12 | -0.37 | 0.00 | 120.16 | fixed-line/artifact? | good | -1 | DROP |
| 1 | 991.0 | 17.89-19.89 | 12.31 | 13.92 | 2.85 | 12.47 | 37.90 | walking/structural band | good | 11 | USE |
| 15 | 973.0 | 18.39-20.39 | 3.01 | 3.27 | -0.43 | 1.57 | 37.90 | walking/structural band | good | 3 | DROP |
| 3 | 905.0 | 19.90-21.90 | 12.61 | 17.37 | -0.85 | 10.00 | 32.26 | walking/structural band | good | 10 | USE |
| 16 | 905.0 | 19.45-21.45 | 6.73 | 6.21 | 1.63 | 4.91 | 37.90 | walking/structural band | good | 7 | MAYBE |
| 2 | 886.0 | 18.44-20.44 | 12.75 | 17.37 | 1.17 | 11.59 | 32.26 | walking/structural band | good | 11 | USE |
| 4 | 739.0 | 21.41-23.41 | 13.70 | 19.28 | 2.51 | 13.28 | 43.55 | walking/structural band | good | 11 | USE |
| 17 | 733.0 | 22.16-24.16 | 3.98 | 4.02 | -0.45 | 2.04 | 120.16 | fixed-line/artifact? | good | 0 | DROP |
| 5 | 553.0 | 23.42-25.42 | 16.41 | 15.39 | 8.22 | 14.98 | 33.06 | walking/structural band | good | 11 | USE |
| 18 | 469.0 | 24.83-26.83 | 1.69 | 1.91 | -0.39 | 0.99 | 58.87 | fixed-line/artifact? | good | -3 | DROP |
| 6 | 372.0 | 26.33-28.33 | 14.57 | 17.96 | 4.20 | 13.26 | 33.06 | walking/structural band | good | 11 | USE |
| 19 | 267.0 | 27.29-29.29 | 4.33 | 2.73 | -1.46 | 1.74 | 58.87 | fixed-line/artifact? | good | -2 | DROP |
| 7 | 234.0 | 27.89-29.89 | 8.36 | 3.19 | -6.17 | 2.10 | 58.87 | fixed-line/artifact? | good | 1 | DROP |
| 20 | 181.0 | 11.46-13.46 | -0.01 | 1.04 | -2.95 | -0.15 | 58.87 | fixed-line/artifact? | poor | -5 | DROP |
| 8 | 103.0 | 27.89-29.89 | 11.16 | 1.23 | 2.38 | 10.19 | 133.06 | walking/structural band | good | 6 | MAYBE |
| 9 | 103.0 | 27.99-29.99 | 12.13 | 4.19 | 8.63 | 6.14 | 133.06 | walking/structural band | good | 6 | MAYBE |
| 10 | 103.0 | 27.99-29.99 | 7.30 | 3.20 | 1.17 | 5.86 | 138.71 | walking/structural band | good | 4 | MAYBE |
