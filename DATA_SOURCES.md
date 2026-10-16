# Data sources

Synthetic road profiles, simulation outputs and trained weights are included with the experiment code. Their generation settings and splits are recorded in the corresponding protocol and scenario files.

The measured road excerpts come from **Data subset for road condition modelling – reference data**, by Asmus Skar Christiansen, Tommy Sonne Alstrøm, Thea Brüsch, Anders Malherbes Vestergaard, Jakob Larsen, Matteo Pettinari, Shahrzad M. Pour, Ekkart Kindler and Uwe Schlotz.

- Source and DOI: https://doi.org/10.11583/DTU.23097002
- Source license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Original file: `m13_zp_vh.csv`; original MD5: `d1c74b645b169dcd88ce99d71ffbb43e`.
- Channel: P79 right-wheelpath elevation, Laser 21.
- Stored source excerpts and provenance: `results/road_halfcar/source/`, `results/road_coverage/`, and `results/road_coverage_centered/`.

Processing includes selecting road intervals, converting millimetres to metres, removing a linear height trend, and sampling the profile for the specified simulation speed. Exact intervals, transforms and checksums are documented by each study's code and provenance files. These transformations do not change the license or attribution of the source dataset. The CC BY 4.0 license applies to the sourced road data; it is not a new license grant for the entire repository.
