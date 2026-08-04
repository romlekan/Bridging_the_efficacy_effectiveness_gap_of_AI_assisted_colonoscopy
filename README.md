# Bridging the efficacy-effectiveness gap of AI-assisted colonoscopy across community endoscopy centers

This repository contains the cohort-level analysis for the pooled community endoscopy study. It evaluates adenoma detection, implementation fidelity, temporal attenuation, implementation components, RE-AIM dimensions, negative controls, matching, and quantitative bias. The supplied CSV files transcribe the aggregate tables distributed with the manuscript. No patient-level records are included.

## Contents

The `data` directory holds the pooled cohort, outcome, moderator, temporal, and sensitivity tables. The `code/colonoscopy_fidelity` package contains the Python analysis. `code/statistical_analysis.R` provides the R 4.4 analysis corresponding to the mixed-model, GEE, spline, ROC, and sensitivity procedures. `configs/primary.yaml` fixes the statistical thresholds and the optional deep tabular sensitivity workload. Verified public source links are listed in `dataset_links.txt`.

## Installation

The environment is intentionally fixed to Python 3.9, PyTorch 2.0.1, CUDA 11.7, NumPy 1.24.4, pandas 1.5.3, SciPy 1.10.1, scikit-learn 1.2.2, and statsmodels 0.13.5.

```bash
conda env create -f environment.yml
conda activate colonoscopy-fidelity
python -m pip install --no-deps .
```

The container requires an NVIDIA runtime compatible with CUDA 11.7.

```bash
docker build -t colonoscopy-fidelity .
docker run --rm --gpus all -v "$PWD/outputs:/workspace/outputs" colonoscopy-fidelity analyze
```

The R route requires R 4.4 with lme4, geepack, rms, pROC, MatchIt, sandwich, lmtest, splines, boot, and jsonlite.

## Data

The analysis uses 16 anonymized published cohorts comprising 8,247 procedures before harmonized exclusions and 7,854 procedures in the primary pooled population. Eight cohorts document structured support and eight are matched unsupported cohorts. The source material reports aggregate outcomes only. It does not release patient-level features or endoscopic video.

The CSV files are the analysis inputs:

- `cohorts.csv` transcribes Supplementary Table S4.
- `outcomes.csv` transcribes Table 1.
- `moderators.csv` transcribes Table 2.
- `sensitivity.csv` transcribes Table 5.
- `temporal.csv` transcribes Supplementary Tables S7 and S8.

Run the analysis from the repository root:

```bash
colonoscopy-analysis analyze
```

The command writes JSON and CSV artifacts to `outputs`. The R analysis is launched with:

```bash
Rscript code/statistical_analysis.R . outputs
```

## Statistical methods

The primary implementation pools binary adenoma events and fits an aggregate binomial model. The expanded cohort representation supports an exchangeable GEE sensitivity model. Heterogeneity is summarized with Cochran Q, between-cohort variance, and I². Continuous implementation fidelity is analyzed with a three-knot restricted cubic spline at the 25th, 50th, and 75th percentiles. Threshold discrimination uses ROC area and the Youden index. Additional modules provide propensity-score matching, overlap and inverse-probability weighting, temporal decay, E-values, negative controls, leave-one-cohort-out analyses, quantitative bias grids, resampling intervals, diagnostics, implementation economics, and RE-AIM summaries.

The manuscript reports a primary ADR difference of 5.8 percentage points, adjusted odds ratio 1.28 with 95% confidence interval 1.11–1.47, P=0.001, and number needed to treat 17. It reports cohort-level variance 0.013, I² 31%, fidelity correlation 0.71, and a fidelity threshold of 72 with AUROC 0.89. The report stores these targets separately from estimates recalculated from rounded aggregate tables. Rounded cohort percentages cannot recover covariate-adjusted patient-level estimates exactly.

## Deep tabular sensitivity workload

The neural module is isolated from the paper’s primary statistics. It is available for high-dimensional cohort sensitivity experiments when private institutional covariates are supplied under an appropriate governance process. No private data schema, examples, identifiers, or locations are embedded.

The fixed workload uses 32 NVIDIA A100 80 GB GPUs, effective batch size 16,777,216, 48 attention blocks of width 8,192, 128 ensemble members, float64 arithmetic, learning rate 1e-7, and 20,000 epochs. Disk planning assumes 8 TB. This workload is not required to recompute the aggregate manuscript tables.

## Expected outputs

`analysis.json` contains environment metadata, input hashes, pooled estimates, GEE results, heterogeneity, paper targets, RE-AIM summaries, temporal analyses, and quantitative bias results. `cohort_summary.csv` contains group totals. `leave_one_out.csv` records influence of each omitted cohort. `manifest.json` records file sizes and SHA-256 values.

## Scope

The repository does not contain clinical decision support, real-time video processing, model inference, patient identifiers, provider identifiers, institutional identifiers, unpublished source records, or private registry data. Cohort labels follow the anonymized supplementary table. Results are cohort-level associations and do not establish causation.
