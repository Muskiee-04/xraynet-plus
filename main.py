"""
Launch the Streamlit UI. Preferred: `streamlit run app/main.py` from the repository root.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    app_path = root / "app" / "main.py"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
        ]
    )
