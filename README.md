# plainproof

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23647996.svg)](https://doi.org/10.5281/zenodo.23647996)

Archived on Zenodo with the sealed held-out outputs: DOI [10.5281/zenodo.23647996](https://doi.org/10.5281/zenodo.23647996) (v0.2.0, code plus `e0-heldout.tar`, `e0b-heldout.tar` and `SHA256SUMS`); all versions: DOI [10.5281/zenodo.23647649](https://doi.org/10.5281/zenodo.23647649); version 0.1.0 (code only): DOI [10.5281/zenodo.23647650](https://doi.org/10.5281/zenodo.23647650). Also on PyPI: `pip install plainproof`.

Reproduction package for **"How many follow-up questions distinguish genuine understanding from apparent mastery?"** (EDM 2027 submission, simulation study). It asks how many adaptive follow-up questions a classification test needs to tell genuine understanding from apparent mastery at a calibrated confidence target, and how that changes as a learner outsources a share *d* of the follow-ups to an AI. A second, sealed study (E0b) tests whether planted hard probe items separate outsourcing learners from honest ones.

Everything here is CPU-only and needs no paid service. Code: MIT (`LICENSE`). Docs, figures and tables: CC BY 4.0 (`LICENSE-docs.md`).

## From install to the figure (about 10 minutes)

```sh
pip install 'plainproof[test,run]'
plainproof init ~/plainproof        # writes the working directory: sealed scripts, freezes, receipts, docs, tests
cd ~/plainproof
plainproof test                     # 43 tests: SPRT against catIrt 0.5.1's saved steps, exact stopping time, simulator, signatures, probes
plainproof verify                   # every sealed file against both freezes (undoes the documented substitutions, then sha256)
plainproof fetch heldout            # the sealed held-out outputs from Zenodo, hash-checked
plainproof figures                  # tables and figures from the held-out summaries (needs R 4.x on PATH; base graphics only)
```

The figures land in `reports/edm-v2-e0/figures/` and `reports/edm-v2-e0b/figures/`, byte-identical to the published copies. `plainproof init` writes the files in the exact layout the sealed scripts expect; nothing sealed changes. Every other command runs inside that directory, or takes `--root DIR`. The local path inside the sealed `scripts/edm_v2_e0/r_env.sh` is part of the frozen record; the `scripts/rscript.sh` wrapper, which `plainproof figures` calls, is the one to use.

## Full rerun (about 4 hours of CPU)

Only if you want to regenerate everything, including the held-out outputs.

```sh
plainproof fetch bobcat     # BOBCAT (MIT) at its pinned commit, with the preprocessed ASSISTments 2009-2010 file
plainproof fit              # 2PL item fit on the development half of learners; checked against the receipt's hash
plainproof retrain          # BOBCAT checkpoint, about 2 hours (author defaults, fold 1, seed 221), compared with the original run
plainproof run e0           # E0 held-out run and summary; refuses to overwrite an existing cell
plainproof run e0b          # E0b held-out run and summary
```

The ASSISTments terms of use forbid passing the data on, so this package never contains it: `fetch bobcat` downloads it from its source, and you accept those terms yourself. Two caveats, stated plainly. The sealed code checks the item fit's sha256, so a different numpy can stop `fit`; and it asserts that the BOBCAT checkpoint's best epoch is 105, so a retrain that is not bit-identical stops `run e0`. In either case, use the published held-out outputs (`plainproof fetch heldout`). The development stages (`scripts/edm_v2_e0/study.py tune|calibrate|calibrate_sprt_full|pilot`) are included for the record; their outputs (synthetic learners only) are in `data/edm-v2-e0/study/`.

## Fingerprints (sha256)

| What | sha256 |
|---|---|
| E0 sealed freeze `reports/edm-v2-e0/freeze.json` | `a0f8ce404f2b5f7e671e2503a33b4554040e80714f6053553507e0972acdbcdc` |
| E0b sealed freeze `reports/edm-v2-e0b/freeze.json` | `3fa0b8d457d813f59abd5255f9303e44deb2119ee79d6a9b92439a6c1cbf58ed` |
| E0 held-out summary (`e0-heldout.tar` → `heldout/summary.json`) | `aff9a8c49d48821c0e9343cd058c4e0e2f1659ffabcaa719824412f3b14abc90` |
| E0b held-out summary (`e0b-heldout.tar` → `heldout/summary.json`) | `617be427feac24df2e342da00d34e91c78d5de91592cee390ef3516ae9e2ff3f` |
| `e0-heldout.tar` (Zenodo 23647996; also a release asset) | `bd06a11eb7163da102cad6ea0df12ea79e52ff94aab1dc35d28ec3910af2be22` |
| `e0b-heldout.tar` (Zenodo 23647996; also a release asset) | `68219f228c41da972fe4c8f92c218e01f85f08eed864185f237cecda05314bbf` |
| ASSISTments 2009 file at BOBCAT e6b6245e | `c07ca9fa0f08207b06488cec527725938fe96e81e3c1762839e1ad0b84f974cc` |
| 2PL item fit (regenerated in step b) | `12a3b167cd9c9b78b3c3c7faf71b2e464b0943a454f6418faf0f99b1114cb7fc` |
| catIrt 0.5.1 CRAN binary used for the fixture | `f5418f447e91d184791e668bd2a135ea5da40c981d4a76826018295b6ea9f079` |

## What is in here

- `src/plainproof/`: item model, SPRT arm, exact stopping time, simulator (uniform and uncertainty-triggered outsourcing), item fit, signatures, planted probes (with the give-up model), matched-coverage scoring.
- `src/plainproof/cli.py`: the `plainproof` command. `setup.py` packs the working directory into the wheel for `plainproof init`.
- `scripts/edm_v2_e0/`, `scripts/edm_v2_e0b/`: the runners, report builders and R plots. `scripts/published_cat_reference_v1/`: the BOBCAT loading code the runners call.
- `reports/`: both sealed freezes, receipts, figures and tables. `docs/`: the E0 protocol and the pre-registration log (verbatim extract).
- `substitutions.json`: the only differences between this code and the sealed originals (package name, file locations, data paths). `scripts/verify_frozen.py` undoes them and checks every hash. `scripts/edm_v2_e0/r_env.sh` is kept verbatim as the record of the original R runtime; use `scripts/rscript.sh` instead.

## Recorded departures

All are in `docs/PREREG_EDM_V2.md`: the SPRT tuning rule was revised after its first development pass (labelled post hoc); SPRT's calibration map was refitted once on full paths before the freeze; the E0 held-out run exceeded its 2 GiB memory cap (no result depends on it; the runner now enforces the cap); E0b was sealed before its pilot table had been reviewed (no result affected).

## Appendix: the raw script calls

What each `plainproof` command runs, from the working directory (`REPRO_DATA` defaults to `./data`):

```sh
python -m pytest -n auto -q tests                                       # plainproof test
python scripts/verify_frozen.py                                         # plainproof verify
ZENODO_URL=https://zenodo.org/records/23647996/files bash scripts/fetch_heldout.sh   # plainproof fetch heldout (mirror: .../releases/download/v0.2.0)
python scripts/edm_v2_e0/report.py data/edm-v2-e0/heldout/summary.json  # plainproof figures (E0 tables)
bash scripts/rscript.sh scripts/edm_v2_e0/plot_error_vs_d.R reports/edm-v2-e0/error-vs-d.csv reports/edm-v2-e0/figures/error-vs-d.png <E0 summary sha256>
python scripts/edm_v2_e0b/report.py data/edm-v2-e0b/heldout/summary.json
bash scripts/rscript.sh scripts/edm_v2_e0b/plot_probe_auc.R reports/edm-v2-e0b/probe-auc-vs-d.csv reports/edm-v2-e0b/figures/probe-auc-vs-d.png <E0b summary sha256>
bash scripts/bobcat/fetch_bobcat.sh                                     # plainproof fetch bobcat
python scripts/edm_v2_e0/fit_assist2009.py data/edm-v2-e0/item-fit      # plainproof fit
python scripts/bobcat/retrain.py && python scripts/bobcat/check_checkpoint.py   # plainproof retrain
python scripts/edm_v2_e0/heldout.py run data/edm-v2-e0/heldout && python scripts/edm_v2_e0/heldout.py summarize data/edm-v2-e0/heldout   # plainproof run e0
python scripts/edm_v2_e0b/run.py run data/edm-v2-e0b/heldout && python scripts/edm_v2_e0b/run.py summarize data/edm-v2-e0b/heldout       # plainproof run e0b
scripts/rscript.sh scripts/edm_v2_e0/catirt_sprt_steps.R tests/fixtures/edm_v2_catirt_sprt   # regenerate the catIrt fixture (R with catIrt 0.5.1)
```
