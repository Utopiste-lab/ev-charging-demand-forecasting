"""Execute the frozen protocol in separate, resumable stages."""
import subprocess
import sys
from pathlib import Path

if __name__=="__main__":
    here=Path(__file__).resolve().parent
    stages=[("run_architecture_benchmark.py","--smoke-only"),
            ("run_architecture_benchmark.py",),("run_ablation.py",),
            ("run_objective_comparison.py",),("run_linear_probe.py",)]
    for script,*args in stages:
        print("Running",script,*args,flush=True)
        subprocess.run([sys.executable,"-B","-X","utf8",str(here/script),*args],check=True)

