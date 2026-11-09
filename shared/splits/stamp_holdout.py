"""Write the hold-out stamp. Called by the run_*.ps1 scripts before any method."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.splits.evaluation_sets import write_run_stamp


if __name__ == "__main__":
    write_run_stamp()
