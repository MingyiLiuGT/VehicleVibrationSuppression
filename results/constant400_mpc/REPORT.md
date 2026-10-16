# Constant 400 kg plant: 300 vs 400 kg MPC at ±500 N

24 roads, 48 trajectories. Primary window: complete 0–24 s. Positive reduction favors the 400 kg model. No identification or mass switching.

| Road | Window | 300 kg model RMS | 400 kg model RMS | 400 kg reduction (%) | 400 kg better roads |
|---|---|---:|---:|---:|---:|
| sine | whole | 0.064792 | 0.065674 | -1.361 | 2/12 |
| sine | interior | 0.064567 | 0.065530 | -1.492 | 2/12 |
| multisine | whole | 0.223104 | 0.221449 | 0.742 | 10/12 |
| multisine | interior | 0.228525 | 0.226721 | 0.789 | 11/12 |

RMS units: m/s². Interior window is 4–20 s. Additional suspension travel, tire load, saturation and realized multiobjective stage cost are in summary.json and per_road.csv. The realized cost uses actual plant outputs with the unchanged core.MPC weights; it is not a finite-horizon predicted optimal cost.
