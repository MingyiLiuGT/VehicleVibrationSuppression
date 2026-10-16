# MPC online mass identification comparison

Same frozen mass study at ±800 N; no NN or controller retuning. Force normalization and effort penalty are unchanged from ±500 N. Positive reduction means lower acceleration RMS.

| Road | Mass (kg) | Fixed RMS | Adaptive RMS | True mass RMS | Adaptive reduction (%) | True mass reduction (%) | Improved roads |
|---|---:|---:|---:|---:|---:|---:|---:|
| sine | 200 | 0.00644 | 0.00644 | 0.00407 | 0.00 | 36.75 | 0/12 |
| sine | 240 | 0.00496 | 0.00496 | 0.00389 | 0.00 | 21.54 | 0/12 |
| sine | 360 | 0.00298 | 0.00298 | 0.00356 | 0.00 | -19.55 | 0/12 |
| sine | 420 | 0.00257 | 0.00257 | 0.00354 | 0.00 | -37.81 | 0/12 |
| multisine | 200 | 0.09925 | 0.10102 | 0.10093 | -1.78 | -1.69 | 5/12 |
| multisine | 240 | 0.08866 | 0.08961 | 0.08975 | -1.07 | -1.23 | 4/12 |
| multisine | 360 | 0.07029 | 0.06854 | 0.06882 | 2.50 | 2.09 | 9/12 |
| multisine | 420 | 0.06540 | 0.06195 | 0.06252 | 5.28 | 4.41 | 10/12 |

RMS unit: m/s². Primary window: 4–20 s (160 m at 10 m/s). All cases retained. Early 4–8 s, later 8–20 s, full and recovery windows are in summary.json and per_road.csv.
Figure case: sine road 16007, 200 kg; chosen by upper-median fixed MPC RMS, not adaptive gain.
All three controllers were simulated at ±800 N; no ±500 N reference trajectories were reused. All new trajectories are under traces/.
The RLS tuning is inherited from the NN study. Results assess transferring that identifier to MPC; they do not establish the best achievable MPC adaptation.
