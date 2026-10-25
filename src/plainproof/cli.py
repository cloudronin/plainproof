"""plainproof: the command-line front end of this reproduction package.

It runs the repository's own scripts, so it needs a checkout or the unpacked sdist (the sealed scripts locate the
repository by its layout). Pass --root, or run it from the repository root.

  plainproof fetch [bobcat|heldout|all]   fetch BOBCAT + the ASSISTments file (hash-checked), and/or the Zenodo outputs
  plainproof fit                          refit the 2PL item parameters on ASSISTments development learners
  plainproof run e0|e0b                   held-out run and summary (refuses to overwrite an existing cell)
  plainproof figures                      rebuild tables and figures from the held-out summaries (needs R for the plots)
  plainproof verify                       check every sealed file against both freezes
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


def find_root(given):
    root = Path(given or os.getcwd()).resolve()
    if not (root / 'substitutions.json').exists() or not (root / 'reports' / 'edm-v2-e0' / 'freeze.json').exists():
        raise SystemExit(f'{root} is not a plainproof checkout (no substitutions.json or reports/); use --root')
    return root


def sh(root, *cmd):
    env = dict(os.environ)
    env.setdefault('REPRO_DATA', str(root / 'data'))
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=root, env=env)
    if r.returncode:
        raise SystemExit(r.returncode)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog='plainproof', description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', help='repository root (default: current directory)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch'); f.add_argument('what', nargs='?', default='all', choices=('bobcat', 'heldout', 'all'))
    sub.add_parser('fit')
    r = sub.add_parser('run'); r.add_argument('study', choices=('e0', 'e0b'))
    sub.add_parser('figures')
    sub.add_parser('verify')
    a = ap.parse_args(argv)
    root = find_root(a.root)
    py = sys.executable
    data = Path(os.environ.get('REPRO_DATA', root / 'data'))
    if a.cmd == 'fetch':
        if a.what in ('bobcat', 'all'):
            sh(root, 'bash', 'scripts/bobcat/fetch_bobcat.sh')
        if a.what in ('heldout', 'all'):
            if not os.environ.get('ZENODO_URL'):
                raise SystemExit('set ZENODO_URL to the Zenodo record file URL to fetch the held-out outputs')
            sh(root, 'bash', 'scripts/fetch_heldout.sh')
    elif a.cmd == 'fit':
        sh(root, py, 'scripts/edm_v2_e0/fit_assist2009.py', data / 'edm-v2-e0' / 'item-fit')
    elif a.cmd == 'run':
        if a.study == 'e0':
            out = data / 'edm-v2-e0' / 'heldout'
            sh(root, py, 'scripts/edm_v2_e0/heldout.py', 'run', out)
            sh(root, py, 'scripts/edm_v2_e0/heldout.py', 'summarize', out)
        else:
            out = data / 'edm-v2-e0b' / 'heldout'
            sh(root, py, 'scripts/edm_v2_e0b/run.py', 'run', out)
            sh(root, py, 'scripts/edm_v2_e0b/run.py', 'summarize', out)
    elif a.cmd == 'figures':
        s0, s1 = data / 'edm-v2-e0' / 'heldout' / 'summary.json', data / 'edm-v2-e0b' / 'heldout' / 'summary.json'
        sh(root, py, 'scripts/edm_v2_e0/report.py', s0)
        sh(root, 'bash', 'scripts/rscript.sh', 'scripts/edm_v2_e0/plot_error_vs_d.R', 'reports/edm-v2-e0/error-vs-d.csv',
           'reports/edm-v2-e0/figures/error-vs-d.png', sha(s0))
        sh(root, py, 'scripts/edm_v2_e0b/report.py', s1)
        sh(root, 'bash', 'scripts/rscript.sh', 'scripts/edm_v2_e0b/plot_probe_auc.R', 'reports/edm-v2-e0b/probe-auc-vs-d.csv',
           'reports/edm-v2-e0b/figures/probe-auc-vs-d.png', sha(s1))
    elif a.cmd == 'verify':
        sh(root, py, 'scripts/verify_frozen.py')


if __name__ == '__main__':
    main()
