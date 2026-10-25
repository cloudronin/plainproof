# EDM 2027 v2, E0 protocol draft (not frozen)

Draft, 24 October 2026, revised with Vishnu's rulings of the same day. Not frozen; nothing is pushed.
Paper scope: Google Doc `1kO2OZ_NReH_3AN9ta5TW_yz9pMYrNn68DKz_vbJ2MBs`, tab "v2: narrowed scope".
Branch `cloudronin/edm-v2-e0`. Nothing below has been run on held-out learners.

## What E0 built and verified

| Piece | State | Evidence |
|---|---|---|
| SPRT classification arm | Wald SPRT at one cut, information-at-cut selection, truncation as its own outcome | `src/plainproof/sprt.py` |
| catIrt replay | All 369 saved steps of catIrt 0.5.1 (25 simulees, 40-item bank) replayed: same items, log ratios within 1e-9, same decisions (19 classified, 6 truncated) | `tests/test_edm_v2_e0.py::test_sprt_replays_catirt_steps`; fixture from `scripts/edm_v2_e0/catirt_sprt_steps.R` |
| Exact stopping time | Finite-horizon calculation for identical items with truncation; matches full enumeration to 1e-12; simulator matches it within Monte Carlo error at d = 0 and 0.5, AI accuracy 0.95 and 0.75 | `edm_v2/stopping.py`; four parametrised tests |
| Wald approximation | Reported beside the exact value; within 5% of exact when steps are small and truncation is negligible | `stopping.wald`; test |
| Simulator | Understanding from theta only (understands / gap / borderline); disjoint homework, follow-up and transfer banks per family; held-out families transfer-only; uniform and uncertainty-triggered outsourcing on one dial; AI accuracy a parameter | `edm_v2/simulator.py` |
| Leak control | At d = 1, SPRT decisions do not differ between learners who understand and learners with a gap, for both outsourcing models and both AI accuracies | `test_full_outsourcing_carries_no_signal` |
| Item fit | 2PL on ASSIST2009 development learners (hashed-ID half): 156 items with at least 50 development learners, 655 learners, 25 EM rounds. a median 1.15 (0.62 to 2.33); b median -0.31 (-2.71 to 1.42) | private `edm-v2-e0/item-fit/assist2009-dev-2pl.receipt.json`, output sha256 `12a3b167...` |
| catIrt install | catIrt 0.5.1 binary from CRAN into the existing native R library (owner-approved); sha256 recorded | private `edm-v2-e0/catirt-install-receipt.txt` |

26 offline E0 tests pass. The full offline suite passes 468 of 468 (`-n auto`) with ripgrep 15.2.0 installed via Homebrew (the site check needs `rg`; earlier 467/468 was that missing tool). No model or provider calls. Spend $0.

## Settings proposed for approval

Each value has a reason. Values marked (?) are the ones I'm least sure of.

| Setting | Proposed | Reason |
|---|---|---|
| Item bank | The 156 fitted ASSIST2009 items, keeping their real q_ids | Information per item comes from real responses; BOBCAT is trained on the same dataset, so it can select among the same q_ids |
| Families | Synthetic, by ruling: the 156 fitted ASSIST2009 items are grouped into 12 families of 13 by difficulty-stratified random assignment, and 2 families are held out for transfer only | The ASSIST2009 file has no skill tags, so these families are not real skills. The paper says so plainly. Eedi construct tags are an investigation item only (below) |
| Guessing c | 0 in the main analysis; 0.25 as the sensitivity check (ruling) | c is not identified by the fit; 0.25 is the chance rate of a four-option item |
| Bank split per family | 40% homework, 40% follow-up, 20% transfer | Keeps the homework-grade baseline blind to follow-up items |
| Cut score | theta = 0 (population median) | Puts half the population on each side; can move after a pilot on development learners only |
| Borderline margin | 0.25 on each side of the cut | About 20% of learners borderline under N(0, 1); reported separately, never forced |
| SPRT indifference delta | 0.3 | Standard range 0.2 to 0.5 in the classification-testing literature |
| SPRT alpha, beta | 0.05, 0.05 | Conventional; matches the primary 90% confidence target in spirit, not by construction |
| Maximum follow-ups | 20 | About a third of a family's follow-up items; above the 1/3/5/10 checkpoints |
| Checkpoints | 1, 3, 5, 10 (fixed budget) | As in the paper sketch |
| Confidence target | 0.90 primary; 0.80 and 0.95 secondary | As in the paper sketch |
| Dial values d | 0, 0.25, 0.5, 0.75, 1 | As in the paper sketch |
| Outsourcing models | uniform; uncertainty-triggered with g(p) = 4p(1 - p), scaled per learner so the mean share equals d | Vishnu's addition; g peaks at p = 0.5 |
| AI accuracy | 0.95, 0.75 | Vishnu's addition |
| Homework outsourcing h | 0.3 main; 0 and 0.6 as checks (approved) | Sets how many apparent-mastery learners exist |
| Cohort | Homework score at or above 0.8, chosen without truth | "Comparable homework success" |
| Learners | 1,000 development (stopping-rule fitting, calibration, pilot SDs); 2,000 held-out per cell | Precision estimate below |
| Worlds | Matched across policies by seed (same learners, same answers where the same item is asked) | Paired comparisons |

## Endpoints and analysis

- **Primary:** questions needed to reach the 0.90 target at matched decision coverage, per policy, per cell (outsourcing model x AI accuracy x d). Inconclusive learners are kept and counted; a target not reached by 20 questions is reported as unreached.
- **Secondary:** false-mastery and false-gap rates, decision coverage, error-against-coverage curves at each checkpoint, calibration of the frozen stopping rule on the runs it actually stopped, and the added prediction of the unaided transfer score beyond homework score.
- **Intervals:** learner-level bootstrap (2,000 resamples), paired across policies.
- **Confirmatory contrasts (ruling):** the four SPRT-against-fixed contrasts (d = 0 and 0.5, both outsourcing models, AI accuracy 0.95) are confirmatory. The four SPRT-against-BOBCAT contrasts at the same settings are confirmatory only if the BOBCAT gate below passes before the freeze; then Holm runs across all 8. If the gate fails, Holm runs across the 4, the BOBCAT contrasts are exploratory, and the PREREG entry says why. Everything else is exploratory and labelled so.
- **Main threat, stated plainly:** absolute question counts depend on the generator. The headline is relative efficiency between policies, with sensitivity to c, h, the cut, delta and the family grouping.

## Precision estimate (?)

Paired per-learner difference in questions needed, held-out n = 2,000 per cell. If the per-learner SD of the difference is about 6 questions, the standard error is about 0.13 questions, and a difference of about 0.4 questions is detectable at 80% power. The SD is a guess: the first development pilot measures it, and n is reset (development only) before freezing.

## Resources

CPU only, no models, no paid calls. Estimate: 20 cells x 6 policies x 2,000 learners x at most 20 steps, about 4.8 million steps; under an hour in total. Proposed cap: 1,800 s and 2 GiB RSS per cell, 256 MiB new evidence, under the existing ledger and the shared Mac lock.

## BOBCAT gate: PASSED (25 Oct 2026, development learners only)

`scripts/edm_v2_e0/bobcat_gate.py`; report in private `edm-v2-e0/bobcat-gate/bobcat-gate.json`. BOBCAT's epoch-105 checkpoint ran end to end on the 156-item bank (50 follow-up, 50 homework, 56 transfer items; bank seed 2026102500, c = 0): 200 development learners per cell, at d = 0 uniform and d = 0.5 uncertainty, AI accuracy 0.95. Every learner completed 20 steps, no action left the follow-up bank, the weights were unchanged (sha256 checked before and after), and a p_mastery (logistic map of BOBCAT's theta, fitted on even development IDs) was scored by `study/reliability.py` on odd IDs. Mechanical check only. 6.8 s, under the Mac lock, $0. By the ruling, the four SPRT-against-BOBCAT contrasts stay eligible for the confirmatory set.

## Development pilot (25 Oct 2026): two design problems before the freeze

`scripts/edm_v2_e0/pilot.py`; report in private `edm-v2-e0/pilot/pilot.json`. 500 development learners per confirmatory cell; all policies read one shared answer table, so they are paired. Questions needed = first count at which the policy's confidence reaches 0.90, up to 20; unreached recorded as 21.

| Cell | SPRT mean (unreached) | Fixed | Random | BOBCAT | SD of paired SPRT - fixed | SD of paired SPRT - BOBCAT |
|---|---|---|---|---|---|---|
| uniform, d = 0 | 11.9 (127) | 8.2 (81) | 10.2 (106) | 7.8 (44) | 6.8 | 6.9 |
| uniform, d = 0.5 | 13.9 (162) | 7.2 (58) | 9.4 (82) | 13.8 (163) | 7.2 | 8.2 |
| uncertainty, d = 0 | 11.5 (113) | 11.2 (113) | 10.2 (102) | 5.4 (34) | 6.5 | 7.3 |
| uncertainty, d = 0.5 | 12.4 (124) | 10.2 (101) | 10.1 (115) | 13.1 (170) | 8.0 | 9.5 |

These are development numbers for sizing only, not results. What they show:

1. **The policies' confidences are on different scales, so "reaches 0.90" means different things.** SPRT stops on its log ratio at alpha = beta = 0.05; fixed and random use the exact posterior under the true item model; BOBCAT uses a logistic map of its theta. The two d = 0 cells should behave alike, and SPRT and random do, but fixed (8.2 against 11.2) and BOBCAT (7.8 against 5.4) do not: fixed depends on which single random order the seed drew, and BOBCAT's map is refitted per cell on about 230 learners. Proposed fix: calibrate every policy's p_mastery per step on development learners with one method (isotonic, on a calibration split), and read the 0.90 target only from calibrated values. Use several fixed forms (for example 10 seeded orders) and report their spread, not one.
2. **SPRT leaves 23% to 32% unreached by 20 questions,** with delta = 0.3. Proposed fix: choose delta and the maximum on development learners only (for example delta 0.5, maximum 30) before freezing, and report unreached counts in every table.
3. **SPRT's item order is not adaptive.** With one cut, information at the cut gives every learner the same order, so its only adaptive part is when it stops. "SPRT against fixed" therefore compares the cut-score order plus the stopping rule against a fixed random order. The paper should say this.
4. **Sizing:** the worst paired SD is 9.5 questions, so detecting 0.4 questions at 80% power with alpha 0.05/8 needs about 7,200 held-out learners per cell, not 2,000. CPU cost stays small (the pilot ran 2,000 learner-policy paths per cell in under 7 s). Censoring at 21 makes this an approximation.

## Eedi licence (investigation only, nothing downloaded)

- No public page found that names a licence for the NeurIPS 2020 "Diagnostic Questions" data (the one with construct and subject tags).
- The organisers' guide (arXiv 2007.12061) says the data downloads from the CodaLab competition page, and that "the question images have been shared solely for the purpose of this competition and must not be used for any other purpose." So the terms sit behind CodaLab registration, and at least the images are competition-only.
- The arXiv licence covers the paper, not the data.
- Reading: not cleared for this paper. Clearing it needs the CodaLab terms as accepted at registration, or written permission from Eedi. The local Eedi research allowance does not extend to this use without that.

## Rulings recorded (24 Oct 2026)

1. Families: synthetic from the ASSIST2009 item fit, stated plainly. Eedi is an investigation item only; report before any download (done above).
2. Guessing: 0 main, 0.25 check.
3. Homework outsourcing: 0.3 main, 0 and 0.6 checks.
4. Confirmatory set: the 4 SPRT-against-fixed contrasts; the 4 SPRT-against-BOBCAT contrasts only if the BOBCAT gate passes before the freeze.
5. BOBCAT end-to-end run on the 156-item bank is a hard gate before the freeze.

## What each method adapts (stated for the paper)

SPRT adapts **stopping**: with one cut, information at the cut gives every learner the same question order, and only the stopping point depends on the answers. BOBCAT adapts **selection**: its learned policy picks each next question from the learner's answers so far. Fixed and random adapt neither.

## Second development pass (25 Oct 2026, rulings of the same day)

`scripts/edm_v2_e0/study.py` (stages tune, calibrate, pilot); outputs in private `edm-v2-e0/study/`. Each stage used its own development seeds.

- **SPRT width and maximum (development tuning, 500 learners per cell):** chosen width 0.7, maximum 20. Rule changed after the first tuning pass, labelled as such: the first rule required decision error at or below 0.10 in all four cells, and no setting met it, because outsourcing at d = 0.5 raises SPRT error to 0.11 to 0.27 for every setting. That rise is the study outcome, so tuning now uses the two d = 0 cells only (error 0.04 to 0.05, mean 5.1 questions, undecided 4% at width 0.7, maximum 20).
- **Calibration:** one isotonic map per method, fitted once on the same 1,000 development learners (250 per confirmatory cell, borderline excluded), pooled over cells and steps. Saved with its sha256 in `calibration.receipt.json`.
- **Fixed forms:** 10 seeded orders, seeds 2026102600 to 2026102609.
- **Undecided learners** count at the maximum length (20) and are reported as an outcome.
- **Pilot (500 learners per cell):** worst paired SD 8.8 questions (sprt minus fixed, sprt minus BOBCAT, and per fixed form), which gives about 6,200 held-out learners per cell for 0.4 questions at 80% power with alpha 0.05/16. The ruling sets about 7,200 per cell, which covers it.

| Cell | SPRT | Fixed (mean of 10) | Random | BOBCAT | Undecided SPRT / fixed / random / BOBCAT |
|---|---|---|---|---|---|
| uniform, d = 0 | 5.1 | 11.7 | 11.9 | 10.0 | 4% / 36% / 34% / 30% |
| uniform, d = 0.5 | 5.4 | 14.9 | 15.0 | 12.6 | 3% / 46% / 48% / 34% |
| uncertainty, d = 0 | 4.8 | 11.6 | 11.9 | 10.5 | 3% / 33% / 34% / 29% |
| uncertainty, d = 0.5 | 4.9 | 14.8 | 14.8 | 12.8 | 2% / 44% / 44% / 33% |

Development numbers for sizing only, not results.

## Third development pass: one scoring rule for every method (ruling 25 Oct 2026)

Same calibration (unchanged, sha256 in `calibration.receipt.json`), same 500 development learners per cell. Every method is now scored the same way: questions needed = first step on its own path whose calibrated confidence reaches 0.90; never reached = undecided, counted at 20. SPRT's native stop is reported beside it. Error = realised decision error among decided, non-borderline learners.

| Cell | Method | Questions | Error | Undecided |
|---|---|---|---|---|
| uniform, d = 0 | SPRT | 11.6 | 1.5% | 49% |
| | fixed (mean of 10 forms) | 11.7 | 6.4% | 36% |
| | random | 11.9 | 2.8% | 34% |
| | BOBCAT | 10.0 | 7.3% | 30% |
| | SPRT native stop | 5.1 | 2.6% | 4% |
| uniform, d = 0.5 | SPRT | 16.3 | 7.5% | 77% |
| | fixed | 14.9 | 9.6% | 46% |
| | random | 15.0 | 5.6% | 48% |
| | BOBCAT | 12.6 | 16.9% | 34% |
| | SPRT native stop | 5.4 | 27.4% | 3% |
| uncertainty, d = 0 | SPRT | 11.6 | 6.5% | 49% |
| | fixed | 11.6 | 7.0% | 33% |
| | random | 11.9 | 5.5% | 34% |
| | BOBCAT | 10.5 | 7.5% | 29% |
| | SPRT native stop | 4.8 | 3.7% | 3% |
| uncertainty, d = 0.5 | SPRT | 15.3 | 4.5% | 71% |
| | fixed | 14.8 | 8.6% | 44% |
| | random | 14.8 | 5.9% | 44% |
| | BOBCAT | 12.8 | 13.5% | 33% |
| | SPRT native stop | 4.9 | 20.5% | 2% |

Development numbers, not results. Two things they show, for the paper to state:

- SPRT's own boundary stops early (about 5 questions), often before its calibrated confidence reaches 0.90, so under the common rule many SPRT learners are undecided. Its native stop is fast but its error rises to 20% to 27% at d = 0.5.
- The calibration is pooled across cells, so at d = 0.5 a calibrated 0.90 does not deliver 90% correctness for every method (BOBCAT 13.5% to 16.9% error). That is the calibration-under-outsourcing question RQ3 asks; the realised error column carries it.

Variance: worst paired SD 10.0 questions (was 8.8 in pass 2), so the sizing rule now gives about 8,100 learners per cell for 0.4 questions at 80% power, alpha 0.05/16.

## Fourth pass: SPRT runs its fixed order to 20, and its map is refitted (rulings 25 Oct 2026)

- **SPRT path:** the fixed SPRT order now runs every learner to 20 questions. Two points are recorded per learner: where the native boundary fired, and where calibrated confidence first reached 0.90. The common rule scores the crossing; undecided only if it never crosses by 20. The native-stop column is kept beside it with its own error.
- **SPRT map refitted, SPRT only, once:** a map is fitted on the kind of path it scores, and SPRT's was not. Its first map was fitted on paths that stopped at the boundary, so its top calibrated value was 0.878. It could never reach 0.90, and under the common rule SPRT could never decide "understands" (pass-4 check: 54% to 62% of full-path SPRT scores fell outside the fitted range). Refit on the same 1,000 calibration learners (seeds 2026102540 to 43) using their full 20-question paths; 16,200 points, the same learner-steps as the random and BOBCAT maps. The fixed, random and BOBCAT maps are unchanged. Files: `calibration-v2.json` and its receipt, which records both hashes.

### Pilot pass 5 (500 development learners per cell; development numbers, not results)

| Cell | SPRT (common rule) | SPRT native stop | Fixed (10 forms) | Random | BOBCAT |
|---|---|---|---|---|---|
| uniform, d = 0 | 9.5 q, 0.3% err, 19% undecided | 5.1 q, 2.6%, 4% | 11.7 q, 6.4%, 36% | 11.9 q, 2.8%, 34% | 10.0 q, 7.3%, 30% |
| uniform, d = 0.5 | 12.8 q, 9.6%, 29% | 5.4 q, 27.4%, 3% | 14.9 q, 9.6%, 46% | 15.0 q, 5.6%, 48% | 12.6 q, 16.9%, 34% |
| uncertainty, d = 0 | 9.3 q, 2.3%, 19% | 4.8 q, 3.7%, 3% | 11.6 q, 7.0%, 33% | 11.9 q, 5.5%, 34% | 10.5 q, 7.5%, 29% |
| uncertainty, d = 0.5 | 11.5 q, 6.7%, 22% | 4.9 q, 20.5%, 2% | 14.8 q, 8.6%, 44% | 14.8 q, 5.9%, 44% | 12.8 q, 13.5%, 33% |

SPRT scores outside the refitted map's range: 0% in every cell. Worst paired SD 8.9 questions, which gives about 6,300 learners per cell by the sizing rule; the ruling of 8,100 per cell stands and covers it.

## Secondary question: signatures of outsourcing (pre-registered, descriptive)

On the held-out logs. Each per-learner log keeps item id, difficulty, discrimination, bank, family, correctness and the true outsourced flag; arms never see it. Seven statistics (`src/plainproof/signature.py`), each with its expected direction fixed now:

1. difficulty-correctness correlation (expected higher under outsourcing);
2. lz person fit (expected lower);
3. log-likelihood ratio of honest at theta against honest plus uniform outsourcing at d with AI accuracy a, at the honest EAP theta (expected higher; zero at d = 0);
4. homework-minus-follow-up ability gap (expected lower);
5. follow-up-minus-transfer ability gap (expected higher);
6. SPRT confident-and-wrong rate (expected higher). **Simulation-only:** it uses truth, which a deployed system can't see;
7. probe excess: among the probe items the method actually asked in its first k, observed hit rate minus the rate predicted at the learner's EAP theta from those k follow-ups (expected higher). Computed per method on that method's own order, never pooled; NaN when no probe was asked, and the NaN count is reported.

Statistics 1 to 6 use the SPRT order. For each statistic: AUC of honest (the d = 0 cell) against outsourcing (each d > 0 cell), at 5, 10, 15 and 20 follow-ups, both outsourcing models, both AI accuracies. Separation line AUC 0.75, fixed now. Descriptive; no selection among the seven. The primary comparisons, the learner count and the calibration are unchanged by this.

**Probe items**, fixed before the freeze: the 10 hardest follow-up items by fitted difficulty, q_ids 5494, 2522, 5615, 5475, 5447, 14873, 14754, 14792, 8807, 5625 (b 1.42 down to 0.13; ASSIST2009 is an easy bank, so "hardest" is relative). They are ordinary items for selection, stopping, calibration and every primary comparison.

- Both outsourcing models are expected to hand probes to the AI. Under uniform outsourcing they go at rate d, like any item. Under the uncertainty-triggered model they go slightly more often than average (computed on the bank with theta from N(0, 1): 0.56 against 0.50 at d = 0.5), but unevenly: learners with theta below -0.5 outsource them at about 0.39, those above 0.5 at about 0.78, because the model outsources most where the learner's own chance is near 0.5.
- SPRT's fixed order reaches only 1 of the 10 probes in its first 20 questions, so statistic 7 for SPRT rests on at most one item. That is a property of the method, reported as such.

## Freeze candidate

`reports/edm-v2-e0/freeze-candidate.json` holds the hashes of 17 code files, the item fit, the bank roles, the calibration, the tuning record and the pilot, plus every setting, the 20 held-out cells (2 outsourcing models x 2 AI accuracies x 5 dial values) and their seeds. Held-out seeds are 2026110000 + 100 x cell index, spaced so the random-order stream (seed + 7) never reuses another cell's seed. No held-out learner has been simulated. The confirmatory set is the 8 contrasts (SPRT against fixed and against BOBCAT, d = 0 and 0.5, both models, AI accuracy 0.95), Holm across 8, because the BOBCAT gate passed.

## Still open before the freeze

- n per cell: 8,100, approved 25 Oct 2026.
- The freeze itself: the freeze file and its PREREG entry, ready for Vishnu's word to seal.
