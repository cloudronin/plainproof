"""Build hook: copy the reproduction working directory into the package as data, so `plainproof init` can write it
out after `pip install plainproof` alone. The copied files are the repository's own files, unchanged."""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

PAYLOAD = ['scripts', 'reports', 'docs', 'tests', 'data/edm-v2-e0/study', 'substitutions.json', 'README.md', 'LICENSE',
           'LICENSE-docs.md', 'CITATION.cff', 'requirements.txt']


class with_payload(build_py):
    def run(self):
        super().run()
        here = Path(__file__).resolve().parent
        dest = Path(self.build_lib) / 'plainproof' / 'payload'
        if dest.exists():
            shutil.rmtree(dest)
        for item in PAYLOAD:
            src = here / item
            if not src.exists():
                raise SystemExit(f'payload item missing from the source tree: {item}')
            if src.is_dir():
                shutil.copytree(src, dest / item, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                (dest / item).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / item)


setup(cmdclass={'build_py': with_payload})
