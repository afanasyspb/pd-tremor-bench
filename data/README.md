# Data

`tremordb_window_features.csv` is the window-level feature table used in the paper:
one row per 2.0 s window (1.0 s hop) of rest-tremor velocity, 6627 windows from
15 subjects, with 10 features and the metadata columns
`record, subject, hand, condition, dbs_state, med_state, mins_after_dbs_stop,
tremor_group, fs_hz, unit, window_start_s, window_size_s, target_label`.
`target_label` is 1 for DBS OFF and 0 for DBS ON.

The table is derived from the public PhysioNet database
"Effect of Deep Brain Stimulation on Parkinsonian Tremor" (Beuter, Titcombe, Glass; 2001):
https://physionet.org/content/tremordb/1.0.0/

The raw recordings are needed only to rebuild the feature table or to run the
raw-signal CNN baseline. Download them into `data/tremordb/`:

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/tremordb https://physionet.org/files/tremordb/1.0.0/
```

Then rebuild the feature table (optional, the committed file is identical):

```bash
python extract_features.py --data-dir data/tremordb --out data/tremordb_window_features.csv
```

Please cite the dataset and PhysioNet when using this data:

- Beuter A, Titcombe MS, Richer F, Gross C, Guehl D. Effect of deep brain stimulation on
  amplitude and frequency characteristics of rest tremor in Parkinson's disease.
  Thalamus & Related Systems 1(3):203-211, 2001.
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new
  research resource for complex physiologic signals. Circulation 101(23):e215-e220, 2000.
