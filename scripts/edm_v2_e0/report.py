"""EDM v2 report pieces from the held-out summary only (never the cell files, never re-simulating).

Writes reports/edm-v2-e0/: error-vs-d.csv (figure data), signature-table.md, results-numbers.json.
Every output names the summary's sha256. Usage: python report.py <summary.json>
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'reports' / 'edm-v2-e0'
METHODS = [('sprt', 'SPRT'), ('fixed', 'Fixed (10 forms)'), ('random', 'Random'), ('bobcat', 'BOBCAT'), ('sprt_native', 'SPRT native stop')]
STATS = [('difficulty_correctness', '1 Difficulty-correctness'), ('lz', '2 lz person fit'),
         ('llr_outsourcing', '3 LLR (true d)'), ('hw_minus_fu', '4 Homework - follow-up'),
         ('fu_minus_transfer', '5 Follow-up - transfer'), ('sprt_confident_wrong', '6 SPRT confident-wrong*')]


def main(summary_path):
    raw = Path(summary_path).read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    s = json.loads(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    # (2) Figure data: realised error against d, AI accuracy 0.95, both models.
    lines = ['model,d,method,error_pct,questions,undecided_pct']
    for r in s['table']:
        if r['a_ai'] != 0.95:
            continue
        for key, label in METHODS:
            v = r[key]
            lines.append(f"{r['model']},{r['d']},{label},{100 * v['error']:.3f},{v['questions']:.3f},{100 * v['undecided']:.2f}")
    (OUT / 'error-vs-d.csv').write_text('\n'.join(lines) + '\n')
    # (3) Signature table.
    md = [f'# EDM v2 held-out: outsourcing signature AUCs', '',
          f'Source: held-out `summary.json`, sha256 `{h}`. AUC of the honest cell (d = 0) against each d > 0 cell, '
          'same outsourcing model and AI accuracy; 8,100 learners per cell. Separation line 0.75 (fixed before the run). '
          'Bold = at or above 0.75. Secondary, descriptive; no selection among the seven.', '',
          '**Caption notes.** (a) lz is oriented so that higher AUC means outsourcers misfit more. At AI accuracy 0.95 it '
          'runs the other way (AUC below 0.5 at high d): answers from a 0.95-accurate AI fit the item model well at a high '
          'theta. (b) Statistic 3 (LLR) uses the true d and AI accuracy for its alternative hypothesis, so it is '
          'oracle-informed. (c) * Statistic 6 uses truth: a simulation-only check. (d) Statistic 7 (probe excess) is per '
          'method on its own order; fixed is the mean AUC over the 10 forms; SPRT is NaN at 5 items because its order '
          'asks no probe that early.', '']
    for a in (0.95, 0.75):
        for model in ('uniform', 'uncertainty'):
            md.append(f'## {model} outsourcing, AI accuracy {a}' + (' (exploratory)' if a != 0.95 else ''))
            md.append('')
            md.append('| d | k | ' + ' | '.join(l for _, l in STATS) + ' | 7 Probe: SPRT | Fixed | Random | BOBCAT |')
            md.append('|' + '---|' * (2 + len(STATS) + 4))
            for r in s['signature_auc']:
                if r['a_ai'] != a or r['model'] != model:
                    continue
                f = lambda v: 'NaN' if v != v else (f'**{v:.2f}**' if v >= 0.75 else f'{v:.2f}')
                pe = r['probe_excess']
                md.append(f"| {r['d']} | {r['k']} | " + ' | '.join(f(r[k]) for k, _ in STATS) + ' | '
                          + ' | '.join(f(pe[m]['auc']) for m in ('sprt', 'fixed', 'random', 'bobcat')) + ' |')
            md.append('')
    (OUT / 'signature-table.md').write_text('\n'.join(md))
    # Numbers for the paper text.
    nums = dict(summary_sha256=h, confirmatory=s['confirmatory'], table=s['table'],
                signature_k20_a095=[r for r in s['signature_auc'] if r['k'] == 20 and r['a_ai'] == 0.95])
    (OUT / 'results-numbers.json').write_text(json.dumps(nums, indent=1))
    print(h)


if __name__ == '__main__':
    main(sys.argv[1])
