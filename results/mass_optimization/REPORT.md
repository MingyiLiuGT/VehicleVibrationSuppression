# Mass adaptation and road coverage evidence

The final refined network uses expanded synthetic road coverage and height-invariant first-layer weights. Dynamics, MPC objective, ±500 N limit, 10 ms period, mass interpolation, and online RLS are unchanged. The mass step changes inertia about static equilibrium.

## Changed-mass segment, 4–20 s

Each entry averages all twelve roads. Reduction is computed from group mean RMS, not the average of individual percentage gains.

| Road | Mass kg | Fixed RMS | Adaptive RMS | True-mass RMS | Reduction % | Terminal error % |
|---|---:|---:|---:|---:|---:|---:|
| sine | 200 | 0.12350804 | 0.11514512 | 0.11368190 | 6.771156 | 19.781875 |
| sine | 240 | 0.10430369 | 0.10146210 | 0.10087865 | 2.724346 | 12.385775 |
| sine | 360 | 0.07194747 | 0.07256200 | 0.07293674 | -0.854134 | 9.399555 |
| sine | 420 | 0.06233517 | 0.06309541 | 0.06359655 | -1.219595 | 12.773220 |
| multisine | 200 | 0.38980350 | 0.39090551 | 0.39129322 | -0.282708 | 2.092135 |
| multisine | 240 | 0.33845106 | 0.33969061 | 0.33975820 | -0.366243 | 2.590781 |
| multisine | 360 | 0.24710594 | 0.24625856 | 0.24619244 | 0.342923 | 3.614505 |
| multisine | 420 | 0.21995152 | 0.21838701 | 0.21829482 | 0.711300 | 3.553928 |

The 200 kg sinusoidal ablation gives gains of 3.668955%, 5.162442%, and 6.771156% for the baseline with earlier RLS, refined NN with earlier RLS, and refined NN with tuned RLS. The original RLS uses λ=0.99 and gate=0.05 m/s²; the retained tuned RLS uses λ=0.97 and gate=0.02 m/s². All use 180–480 kg projection bounds.

Representative road 16001 remains the upper median by fixed-controller RMS. Its gain is 14.9930%, with early/terminal mass errors 12.9108%/4.9167%. All 96 road/mass cases and unfavorable outcomes are retained.

## Training and reproducibility

The original paired labels remain split by source scenario. Expanded training adds 48 synthetic roads and 12 validation roads; see ../road_coverage_centered/protocol.json and ../../refine_road_coverage.py. Fixed initial weights are ../road_coverage/initial_policy.npz. Use `python refine_road_coverage.py --centered --stage train` to reproduce the final network. The best of 800 epochs is 591. Synthetic validation, not measured-road performance, chooses the checkpoint. Earlier viewed measured segments are regression evidence; a third nonoverlapping interval is evaluated after freeze. No measured samples enter training or normalization.

## Validation and timing

All 1056 trajectories pass physical state propagation, force balance, causal estimator replay, NN action replay, and RMS checks. Minimum normal load ratio: 0.19200293. Maximum applied force: 500 N. Detailed residuals are in validation.json.

The four software paths are remeasured on Apple M4 with 100 warm-ups and 5,000 calls per path, float64, requested single-thread execution and cyclic garbage collection disabled. Every wall-time sample is retained. Fixed/adaptive MPC-to-NN mean ratios are 3.40/16.21. Adaptive MPC rebuilds 2910 times; initialization mean is 0.44802715 ms. Sensing, reconstruction, communication, and physical actuation are excluded.

- NN_fixed: mean 0.01676717, SD 0.00797663, P95 0.02095800, maximum 0.27629100 ms.
- MPC_fixed: mean 0.05701069, SD 0.01827265, P95 0.08512920, maximum 0.37529100 ms.
- NN_adaptive: mean 0.02032929, SD 0.00707748, P95 0.02420900, maximum 0.14483400 ms.
- MPC_adaptive: mean 0.32948280, SD 0.24448911, P95 0.57238750, maximum 3.15945800 ms.

Old weights, results, figures, and timing records are archived under archive/road_coverage_revision_20261015/before. See code/README.md for the full reproduction order.
