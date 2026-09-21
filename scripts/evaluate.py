"""Comparison table of all policies. Usage: python scripts/evaluate.py [n_episodes]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_marl.evaluate import main

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
