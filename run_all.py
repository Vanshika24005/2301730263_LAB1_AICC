"""Run the full pipeline end to end:  python src/run_all.py"""
import subprocess, sys
from pathlib import Path

here = Path(__file__).parent
for script in ["00_validate_setup.py", "01_generate_dataset.py", "02_eda.py",
               "03_ml_classifier.py", "04_dl_comparison.py", "05_ethics_simulation.py"]:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(here / script)], check=True, cwd=here)
print("\nAll done. See outputs/ for plots and metrics; build the report with "
      "`python report/build_report.py`.")
