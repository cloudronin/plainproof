"""E0b report pieces from the held-out summary only. Writes reports/edm-v2-e0b/: probe-auc-vs-d.csv (figure data),
probe-table.md, numbers.json. Every output names the summary sha256. Usage: python report.py <summary.json>
"""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'reports' / 'edm-v2-e0b'
MODELS = ('uniform', 'uncertainty', 'giveup')
LABEL = dict(uniform='uniform', uncertainty='uncertainty-triggered', giveup='give-up (exploratory)')


def main(path):
    raw = Path(path).read_bytes(); h = hashlib.sha256(raw).hexdigest(); s = json.loads(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = s['rows']
    lines = ['model,d,raw_hits_auc,probe_share']
    for m in MODELS:
        lines.append(f'{m},0,0.5,0')                      # d = 0 is the honest reference itself
        for r in rows:
            if r['model'] == m and r['a_ai'] == 0.95 and r['k'] == 15:
                lines.append(f"{m},{r['d']},{r['hits']['sprt']['all']:.4f},{r['probe_outsourced_share']:.4f}")
    (OUT / 'probe-auc-vs-d.csv').write_text('\n'.join(lines) + '\n')
    f = lambda v: 'NaN' if v != v else (f'**{v:.2f}**' if v >= 0.75 else f'{v:.2f}')
    md = ['# EDM v2 E0b held-out: planted-probe AUCs', '',
          f'Source: E0b held-out `summary.json`, sha256 `{h}`; sealed freeze `reports/edm-v2-e0b/freeze.json`. '
          '2,000 learners per cell. AUC of the honest cell (d = 0) against each d > 0 cell, same outsourcing model and AI '
          'accuracy. k counts every item asked (probes at positions 4, 9 and 14), so k = 5, 10, 15 and 20 include 1, 2, 3 '
          'and 3 probes. Raw hits are the same for every method (shared probes and answers); probe excess is per method. '
          'Bold = at or above the pre-registered 0.75 line, which applies to the uniform model at d = 0.5, AI accuracy '
          '0.95, k = 15.', '',
          '**Caption notes.** The uncertainty-triggered model\'s null is by construction: it outsources where the '
          'learner\'s own chance is near 0.5, and on the probes it is near 0, so few probes are outsourced (the share column). '
          'The give-up model is exploratory: it outsources most where the own chance is lowest, so probes are outsourced '
          'far above the bank average d.', '']
    for a in (0.95, 0.75):
        for m in MODELS:
            md += [f'## {LABEL[m]}, AI accuracy {a}' + (' (exploratory accuracy)' if a != 0.95 else ''), '',
                   '| d | k | Probes outsourced | Raw hits | Excess SPRT | Excess fixed | Excess random | Excess BOBCAT |',
                   '|---|---|---|---|---|---|---|---|']
            for r in rows:
                if r['model'] == m and r['a_ai'] == a:
                    e = r['excess']
                    md.append(f"| {r['d']} | {r['k']} | {100 * r['probe_outsourced_share']:.0f}% | {f(r['hits']['sprt']['all'])} | "
                              + ' | '.join(f(e[x]['all']) for x in ('sprt', 'fixed', 'random', 'bobcat')) + ' |')
            md.append('')
    (OUT / 'probe-table.md').write_text('\n'.join(md))
    (OUT / 'numbers.json').write_text(json.dumps(dict(summary_sha256=h, at_line=s['at_line'], crossing=s['crossing'],
                                                      k15=[r for r in rows if r['k'] == 15]), indent=1))
    print(h)


if __name__ == '__main__':
    main(sys.argv[1])
