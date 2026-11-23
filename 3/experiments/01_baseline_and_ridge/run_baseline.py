"""Use the unified experiment for official Seasonal Naive and Ridge outputs."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runner = Path(__file__).resolve().parents[1] / "03_tree_models_1h" / "run_comparison.py"
    # Run as a script so its sibling reporting module can be imported normally.
    import sys
    sys.path.insert(0, str(runner.parent))
    runpy.run_path(str(runner), run_name="__main__")
