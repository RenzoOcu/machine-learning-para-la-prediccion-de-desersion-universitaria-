import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_PROJECT_ROOT)

from streamlit.starlette import App  # noqa: E402

# Vercel carga `app` (variable de nivel superior) como app ASGI.
# Requiere streamlit>=1.53 (ver requirements.txt). El script_path es
# absoluto y el CWD se fija a la raíz del proyecto para que app.py encuentre
# los artefactos (modelo_edupredict.pkl, preprocessor_edupredict.pkl, ...).
app = App(str(_PROJECT_ROOT / "app.py"))