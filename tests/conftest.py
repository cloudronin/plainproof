import sys
from pathlib import Path

try:  # prefer the installed package (pip install plainproof); fall back to the source tree
    import plainproof  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
