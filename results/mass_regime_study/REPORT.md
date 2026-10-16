# Focused mass adaptation validation

The frozen NN and causal RLS are unchanged. The experiment selects operating regions, not individual favorable validation trajectories. Main manuscript claims apply only to the declared periodic excitation ranges.

## Design

- Development: 468 mass-only, 3,900 coarse excitation and 2,520 neighborhood NN trajectories. All results retained.
- Decrease range: 19–21 mm amplitude, 1.3–1.5 Hz; primary mass 200 kg, neighbors 220/240 kg.
- Increase range: 24–26 mm amplitude, 0.6–0.8 Hz; primary mass 420 kg, neighbors 360/400 kg.
- Each range uses 24 fresh roads, uniform amplitude/frequency draws and independent phase/noise; cases and primary masses were frozen before validation.
- Reference mass 300 kg; changed interval 4–20 s in a 24 s run. Same physical plant and road for every controller comparison.
- Three NN modes and two MPC modes at six masses: 720 retained traces. No outcome-based time-window, observation or road removal.
- Primary metric: 100 × (1 − mean adaptive road RMS / mean fixed road RMS), using the full 4–20 s interval.
- Figures 7/8 use all-road means. Figure 7 shading is sample SD across 24 estimates. Figure 8 averages the per-road cumulative RMS, without smoothing.

## Validation results

| Group | Fixed NN RMS | Adaptive NN RMS | Reduction % | Improved / total | True-mass MPC gain % |
|---|---:|---:|---:|---:|---:|
| decrease_200 | 0.05114 | 0.02422 | 52.63 | 24/24 | 65.62 |
| decrease_220 | 0.04003 | 0.02332 | 41.74 | 24/24 | 52.20 |
| decrease_240 | 0.03267 | 0.02295 | 29.76 | 24/24 | 36.35 |
| increase_360 | 0.14746 | 0.13001 | 11.84 | 24/24 | 12.50 |
| increase_400 | 0.15938 | 0.13340 | 16.30 | 24/24 | 17.50 |
| increase_420 | 0.16370 | 0.13428 | 17.97 | 24/24 | 19.29 |

RMS units are m/s². The selected ranges give meaningful improvements in both directions and across neighboring masses. This does not reverse the weaker or adverse findings in the original broad-distribution tests; those results remain in `../mass_optimization/`. The two loading directions use different declared excitation ranges, so the difference between their percentages is not a pure mass-direction effect.

## Reproducibility

`protocol.json` records the policy hash, plant, estimator and evaluation definitions. `validation_protocol.json` fixes the regions and road identifiers. CSV files retain all development and validation results; `validation_traces/` contains all 720 traces. `batch_verification.json` checks the vectorized implementation against the original scalar rollout. `validation_audit.json` checks saved dynamics, force balance, policy commands, causal RLS replay, metrics, force limits and tire contact.
