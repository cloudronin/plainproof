"""plainproof: the command-line front end of this reproduction package.

`plainproof init DIR` writes a working directory from the package (sealed scripts, both freezes, receipts,
substitutions.json, R scripts, docs, tests) in the exact layout the sealed scripts expect; nothing sealed changes.
Every other command runs inside that directory, or takes --root DIR.

  plainproof init DIR                     write the working directory (refuses a non-empty DIR)
  plainproof test                         the 43 offline tests
  plainproof verify                       check every sealed file against both freezes
  plainproof fetch heldout|bobcat|all     the sealed held-out outputs, and/or BOBCAT + the ASSISTments file (hash-checked)
  plainproof figures                      rebuild tables and figures from the held-out summaries (needs R for the plots)
  plainproof fit                          refit the 2PL item parameters on ASSISTments development learners
  plainproof retrain                      retrain the BOBCAT checkpoint, then compare it with the original run
  plainproof run e0|e0b                   held-out run and summary (refuses to overwrite an existing cell)
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The sealed held-out outputs, on Zenodo (record 23647996, DOI 10.5281/zenodo.23647996). Their sha256 values are checked
# by scripts/fetch_heldout.sh before use. Fallback mirror: ZENODO_URL=https://github.com/cloudronin/plainproof/releases/download/v0.2.0
HELDOUT_BASE_URL = 'https://zenodo.org/records/23647996/files'
PAYLOAD = Path(__file__).resolve().parent / 'payload'


def find_root(given):
    root = Path(given or os.getcwd()).resolve()
    if not (root / 'substitutions.json').exists() or not (root / 'reports' / 'edm-v2-e0' / 'freeze.json').exists():
        raise SystemExit(f'{root} is not a plainproof working directory; run `plainproof init DIR` and cd there, or pass --root')
    return root


def sh(root, *cmd, env_extra=None):
    env = dict(os.environ)
    env.setdefault('REPRO_DATA', str(root / 'data'))
    env.update(env_extra or {})
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=root, env=env)
    if r.returncode:
        raise SystemExit(r.returncode)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def init(target):
    target = Path(target).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f'{target} is not empty; choose a new directory')
    if not PAYLOAD.exists():
        raise SystemExit('this installation has no payload; reinstall plainproof from the wheel or sdist')
    shutil.copytree(PAYLOAD, target, dirs_exist_ok=True)
    # The sealed scripts import the package from <root>/src and hash its modules there, so the package is written
    # out too, byte for byte from this installation.
    pkg = target / 'src' / 'plainproof'
    pkg.mkdir(parents=True)
    for f in Path(__file__).resolve().parent.glob('*.py'):
        shutil.copy2(f, pkg / f.name)
    for d in ('data/edm-v2-e0', 'data/edm-v2-e0b'):
        (target / d).mkdir(parents=True, exist_ok=True)
    for f in target.rglob('*.sh'):
        f.chmod(0o755)
    print(f'wrote {target}\nnext: cd {target} && plainproof test && plainproof verify && plainproof fetch heldout && plainproof figures')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='plainproof', description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', help='working directory (default: current directory)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('init').add_argument('dir')
    sub.add_parser('test')
    sub.add_parser('verify')
    sub.add_parser('fetch').add_argument('what', choices=('heldout', 'bobcat', 'all'))
    sub.add_parser('figures')
    sub.add_parser('fit')
    sub.add_parser('retrain')
    sub.add_parser('run').add_argument('study', choices=('e0', 'e0b'))
    a = ap.parse_args(argv)
    if a.cmd == 'init':
        return init(a.dir)
    root = find_root(a.root)
    py = sys.executable
    data = Path(os.environ.get('REPRO_DATA', root / 'data'))
    if a.cmd == 'test':
        sh(root, py, '-m', 'pytest', '-n', 'auto', '-q', '-p', 'no:cacheprovider', 'tests')
    elif a.cmd == 'verify':
        sh(root, py, 'scripts/verify_frozen.py')
    elif a.cmd == 'fetch':
        if a.what in ('heldout', 'all'):
            sh(root, 'bash', 'scripts/fetch_heldout.sh', env_extra={'ZENODO_URL': os.environ.get('ZENODO_URL', HELDOUT_BASE_URL)})
        if a.what in ('bobcat', 'all'):
            sh(root, 'bash', 'scripts/bobcat/fetch_bobcat.sh')
    elif a.cmd == 'figures':
        s0, s1 = data / 'edm-v2-e0' / 'heldout' / 'summary.json', data / 'edm-v2-e0b' / 'heldout' / 'summary.json'
        for s in (s0, s1):
            if not s.exists():
                raise SystemExit(f'{s} missing; run `plainproof fetch heldout` first')
        sh(root, py, 'scripts/edm_v2_e0/report.py', s0)
        sh(root, 'bash', 'scripts/rscript.sh', 'scripts/edm_v2_e0/plot_error_vs_d.R', 'reports/edm-v2-e0/error-vs-d.csv',
           'reports/edm-v2-e0/figures/error-vs-d.png', sha(s0))
        sh(root, py, 'scripts/edm_v2_e0b/report.py', s1)
        sh(root, 'bash', 'scripts/rscript.sh', 'scripts/edm_v2_e0b/plot_probe_auc.R', 'reports/edm-v2-e0b/probe-auc-vs-d.csv',
           'reports/edm-v2-e0b/figures/probe-auc-vs-d.png', sha(s1))
    elif a.cmd == 'fit':
        sh(root, py, 'scripts/edm_v2_e0/fit_assist2009.py', data / 'edm-v2-e0' / 'item-fit')
    elif a.cmd == 'retrain':
        sh(root, py, 'scripts/bobcat/retrain.py')
        sh(root, py, 'scripts/bobcat/check_checkpoint.py')
    elif a.cmd == 'run':
        out = data / a.study.replace('e0', 'edm-v2-e0', 1) / 'heldout'
        script = 'scripts/edm_v2_e0/heldout.py' if a.study == 'e0' else 'scripts/edm_v2_e0b/run.py'
        sh(root, py, script, 'run', out)
        sh(root, py, script, 'summarize', out)


if __name__ == '__main__':
    main()
