## T1 features (`src/t1_features.py`)
One row per person from the preprocessed T1, 74 features: for each of 8
Harvard-Oxford subcortical regions (caudate, putamen, pallidum, thalamus; left
and right) the mean CSF/grey/white share from the FAST soft maps and six
first-order statistics of T1 brightness divided by the person's median
white-matter brightness; plus native brain volume and the affine scale.
No shape, texture or regional-volume features (see the module docstring for
why). Subject IDs are pseudonyms; the diagnosis label is never in this table.

```bash
python models/mri/src/t1_features.py   # -> data/processed/features/t1_features.csv
```
