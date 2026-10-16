# MPC online mass identification comparison

Same frozen mass study; no NN or controller retuning. Positive reduction means lower acceleration RMS.

| Road | Mass (kg) | Fixed RMS | Adaptive RMS | True mass RMS | Adaptive reduction (%) | True mass reduction (%) | Improved roads |
|---|---:|---:|---:|---:|---:|---:|---:|
| sine | 200 | 0.12164 | 0.11358 | 0.11191 | 6.63 | 8.00 | 8/12 |
| sine | 240 | 0.10273 | 0.10051 | 0.09945 | 2.17 | 3.20 | 8/12 |
| sine | 360 | 0.07124 | 0.07179 | 0.07202 | -0.77 | -1.10 | 2/12 |
| sine | 420 | 0.06183 | 0.06244 | 0.06283 | -0.98 | -1.61 | 2/12 |
| multisine | 200 | 0.38927 | 0.39034 | 0.39058 | -0.28 | -0.34 | 3/12 |
| multisine | 240 | 0.33805 | 0.33931 | 0.33936 | -0.37 | -0.39 | 2/12 |
| multisine | 360 | 0.24666 | 0.24579 | 0.24575 | 0.35 | 0.37 | 12/12 |
| multisine | 420 | 0.21962 | 0.21805 | 0.21800 | 0.71 | 0.74 | 12/12 |

RMS unit: m/s². Primary window: 4–20 s (160 m at 10 m/s). All cases retained. Early 4–8 s, later 8–20 s, full and recovery windows are in summary.json and per_road.csv.
Figure case: sine road 16001, 200 kg; chosen by upper-median fixed MPC RMS, not adaptive gain.
Fixed/oracle traces reused from ../mass_optimization/mass_heads_final/heldout_traces. Newly simulated adaptive traces are under traces/.
The RLS tuning is inherited from the NN study. Results assess transferring that identifier to MPC; they do not establish the best achievable MPC adaptation.
