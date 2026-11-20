"""Holdout web app: ``streamlit run app/streamlit_app.py`` (needs ``pip install -e ".[app]"``).

The repository root is put on ``sys.path`` so the demo datasets can use the trap
generators in ``bench/`` (not part of the installed package).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holdout.webapp.ui import main  # noqa: E402

main()
