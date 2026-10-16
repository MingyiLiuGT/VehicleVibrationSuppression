# Realized MPC objective comparison

True mass 400 kg throughout; force limit ±500 N; complete 0–24 s, 2400 samples, 12 roads per class. Same physical-output weights for both controllers. These values are realized closed-loop costs, not solver-predicted 30-step objectives.

| Road | Model kg | Acceleration | Travel | Tire | Force | Mean stage cost | Mean 24 s sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| sine | 300 | 0.00847137 | 0.00040620 | 0.00010978 | 0.00267233 | 0.01165968 | 27.98322 |
| sine | 400 | 0.00860826 | 0.00040498 | 0.00011038 | 0.00269187 | 0.01181549 | 28.35718 |
| multisine | 300 | 0.05916150 | 0.00039590 | 0.00019521 | 0.00236208 | 0.06211469 | 149.07527 |
| multisine | 400 | 0.05863493 | 0.00038586 | 0.00019412 | 0.00235149 | 0.06156640 | 147.75937 |

sine: changing model mass from 300 to 400 kg reduces realized cost by -1.3364% (negative means increase).

multisine: changing model mass from 300 to 400 kg reduces realized cost by 0.8827% (negative means increase).
