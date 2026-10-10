import sys
from pathlib import Path

HW2_DIR = Path(__file__).resolve().parents[1]

for path in (HW2_DIR / "third_party" / "DroPE", HW2_DIR / "src", HW2_DIR / "kaggle"):
    sys.path.insert(0, str(path))
