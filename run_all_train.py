"""Lance les trois scripts d'entraînement et capture la sortie."""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "Train folder/train_surrogate_scalar.py",
    "Train folder/ml_surrogate1.py",
    "Train folder/train_surrogate_gp.py",
]

for script in scripts:
    print(f"\n{'='*70}")
    print(f"LANCEMENT : {script}")
    print('='*70)
    sys.stdout.flush()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-u", script],
        cwd=ROOT,
        capture_output=False,
        env=env,
    )
    print(f"\n[Exit code: {result.returncode}]")
    sys.stdout.flush()
    if result.returncode != 0:
        print(f"\nErreur dans {script} — arrêt.")
        sys.exit(result.returncode)

print("\n\nTous les scripts terminés.")
