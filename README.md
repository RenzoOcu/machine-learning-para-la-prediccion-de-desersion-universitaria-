# 🎓 EduPredict Trujillo

**Sistema predictivo de deserción universitaria basado en Machine Learning.**
Proyecto TIF Nivel III — Universidad César Vallejo, Trujillo (2026).

- **Algoritmo principal:** XGBoost + SMOTE (balanceo de clases).
- **Pestañas del dashboard:** 🏠 Inicio · 📝 Predictor Individual · 📊 Análisis de Datos.
- **ODS:** Objetivo 4 "Educación de calidad" — Meta 4.3 (acceso igualitario a la educación superior).

---

## 📚 Dataset oficial (regla innegociable)

El sistema **solo** funciona con el dataset del UCI Machine Learning Repository:

> **"Predict students' dropout and academic success"**
> URL: https://uci-ics-mlr-prod.aws.uci.edu/dataset/697/predict+students+dropout+and+academic+success

- Archivo: `data.csv` (separador `;`).
- **4.424 filas** y **37 columnas** (36 variables predictoras + `Target`).
- Clases del `Target`: `Dropout`, `Graduate`, `Enrolled`.
- El script `train_model.py` **binariza** el target: `Dropout = 1` (clase minoritaria),
  `Graduate/Enrolled = 0`.

**Preprocesamiento:**

| Tipo | Variables | Técnica |
|------|-----------|---------|
| Numéricas (18) | Admission grade, Age at enrollment, Curricular units 1er/2do semestre, Previous qualification (grade), Unemployment rate, Inflation rate, GDP… | `StandardScaler` |
| Categóricas (18) | Marital status, Application mode, Daytime/evening attendance, Nacionality, Course, Gender, Scholarship holder… | `OneHotEncoder` |

> **Nota de ingeniería:** `Previous qualification (grade)` es una **nota numérica
> continua (0–200)** en el dataset UCI real; por eso se **escala**, no se codifica
> en one-hot (una codificación one-hot en una variable continua generaría cientos
> de columnas dummy y dañaría el modelo).

---

## 📁 Estructura del proyecto

```
sistemasInteligentes/
├── data.csv                  <- Descargar del UCI (colocar aquí)
├── train_model.py            <- Script de entrenamiento (XGBoost + SMOTE)
├── app.py                    <- Dashboard web (Streamlit)
├── requirements.txt          <- Dependencias
└── README.md                 <- Este archivo

# Artefactos generados tras entrenar (NO se incluyen en el repo):
├── modelo_edupredict.pkl
├── preprocessor_edupredict.pkl
└── metricas_edupredict.json
```

---

## ⚙️ Instalación paso a paso

### 1. Requisitos previos
- Python **3.9 o superior** (recomendado: 3.10/3.11).
- `pip` actualizado.

### 2. Crear un entorno virtual (recomendado)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Para ejecutar SOLO el dashboard ya entrenado** basta `requirements.txt`.
> **Para reentrenar el modelo** (`train_model.py`) se necesita además `shap` e
> `imbalanced-learn` (SMOTE); instala en su lugar:
> ```bash
> pip install -r requirements-train.txt
> ```

> Si estás en Windows y `xgboost` fallara, instálalo por separado:
> ```powershell
> pip install xgboost
> ```

### 4. Descargar el dataset
1. Ve a la web oficial del UCI (enlace arriba).
2. Descarga el archivo de datos **`data.csv`** (separador `;`).
3. Colócalo en la carpeta del proyecto, junto a `train_model.py`.

---

## 🚀 Paso 1 — Entrenar el modelo

```bash
python train_model.py
```

**Qué hace el script:**
1. Carga `data.csv` con `sep=';'` y valida las 37 columnas reales del UCI.
2. Binariza `Target` (`Dropout=1`, `Graduate/Enrolled=0`).
3. Divide **estratificadamente** en train/test 80/20 (`random_state=42`).
4. Crea un `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`).
5. Aplica **SMOTE solo al conjunto de entrenamiento** dentro de un pipeline de
   `imbalanced-learn` (sin data leakage sobre el test).
6. Entrena `XGBClassifier` (`n_estimators=200`, `learning_rate=0.1`, `max_depth=6`).
7. Evalúa con **Accuracy, Recall, F1-Score y AUC-ROC** e imprime la matriz de confusión.
8. Guarda:
   - `modelo_edupredict.pkl` → clasificador XGBoost.
   - `preprocessor_edupredict.pkl` → ColumnTransformer ajustado.
   - `metricas_edupredict.json` → métricas para el dashboard.

**Salida esperada (aproximada):**
```
Accuracy           : 0.83xx
Recall (Dropout)   : 0.79xx
F1-Score (Dropout) : 0.73xx
AUC-ROC            : 0.88xx
```

---

## 💻 Paso 2 — Ejecutar el dashboard

```bash
streamlit run app.py
```

El navegador se abrirá en `http://localhost:8501`.

**Uso de las pestañas:**

| Pestaña | Descripción |
|---------|-------------|
| 🏠 **Inicio** | Proyecto, metodología, ODS 4.3 y métricas del modelo entrenado. |
| 📝 **Predictor Individual** | Formulario con las variables más influyentes (Admission grade, unidades aprobadas 1er/2do semestre, Scholarship holder, Unemployment rate, Age at enrollment, Gender). Al pulsar **"Predecir"** construye la fila completa de 36 columnas, aplica el preprocesador y usa `model.predict_proba()` para obtener el % de riesgo. Muestra un **gauge Plotly** con codificación por colores: 🔴 >60 % alto · 🟡 30–60 % medio · 🟢 <30 % bajo, con recomendaciones personalizadas. |
| 📊 **Análisis de Datos** | Importancia de características del modelo, KPIs, distribución del target, histogramas, boxplots y matriz de correlación (requiere `data.csv` en la carpeta). |

---

## 🚀 Despliegue en Vercel (Streamlit como ASGI)

El dashboard se sirve en Vercel como una **Vercel Function de Python** que expone
Streamlit mediante su modo ASGI experimental (`streamlit.starlette.App`, Streamlit
**≥ 1.53**). Un solo archivo, `api/index.py`, exporta la app:

```python
# api/index.py
from streamlit.starlette import App
app = App(str(Path(__file__).resolve().parents[1] / "app.py"))
```

**Configuración ya incluida en el repo:**

| Archivo | Propósito |
|---------|-----------|
| `api/index.py` | Entrypoint ASGI (variable `app` de nivel superior). |
| `pyproject.toml` | `[tool.vercel] entrypoint = "api.index:app"` (formato `módulo:variable` que exige Vercel). |
| `requirements.txt` | Solo dependencias de **ejecución** (mantiene el bundle < 500 MB). |
| `vercel.json` | `maxDuration` de la función (Hobby: máx. 300 s). |
| `.vercelignore` | Excluye `.venv`, `__pycache__`, etc. de la subida. |
| `.streamlit/config.toml` | `headless = true` y tema claro. |

> Streamlit usa **WebSockets** (`/_stcore/stream`). Vercel los admite en
> **Functions con Fluid compute** (activado por defecto en proyectos nuevos desde
> abril de 2025, y en *Public Beta* para Python). No hace falta configuración extra.

**Pasos:**

1. Sube el repo a GitHub (incluye `modelo_edupredict.pkl`,
   `preprocessor_edupredict.pkl`, `metricas_edupredict.json` y `data.csv`; ya
   están versionados).
2. En Vercel: **Add New → Project** → importa el repo.
3. Framework: **Other** (el entrypoint se lee de `pyproject.toml`).
4. Build/Install: deja los valores por defecto (Vercel instala
   `requirements.txt`).
5. Deploy. La URL raíz servirá el dashboard.

**Notas:**
- El primer acceso tras un *cold start* puede tardar (arranca
  streamlit + xgboost + plotly + pandas); los siguientes son rápidos.
- Mantén `requirements.txt` **ligero**: si agregas `shap` o `imbalanced-learn`
  al despliegue, el bundle puede superar el límite de 250 MB (500 MB Python/
  *Large Functions*). Esos paquetes viven en `requirements-train.txt`.
- Para depurar localmente el mismo modo ASGI:
  ```bash
  pip install "streamlit[starlette]>=1.53"
  uvicorn api.index:app --host 0.0.0.0 --port 8501
  ```

---

## 🛠️ Solución de problemas

| Problema | Solución |
|----------|----------|
| `No se encontró 'data.csv'` | Descargar el archivo del UCI y colocarlo junto a `train_model.py`. |
| Error al cargar `*.pkl` en el dashboard | Ejecutar primero `python train_model.py` para generar los artefactos. |
| `xgboost` no se instala en Windows | Revisar que tengas Python 64 bits y probar `pip install xgboost --upgrade`. |
| Acentos/caracteres raros en consola | Ejecutar en un terminal con UTF-8 (`chcp 65001` en CMD de Windows). |
| Cambios en el modelo no se reflejan | El dashboard cachea los `.pkl`; reinicia Streamlit (`Ctrl+C` y `streamlit run app.py`). |
| Error de categoría desconocida en predicción | El `OneHotEncoder` usa `handle_unknown="ignore"`, por lo que no debería ocurrir. |

---

## 📜 Licencia y atribución del dataset

- Dataset: *Predict students' dropout and academic success* — UCI Machine Learning Repository
  ([Licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- Autores del dataset: M.V. Martins, D. Tolledo, J. Machado, L.M.T. Baptista, V. Realinho (2019).
- Uso académico-científico para el TIF Nivel III de la UCV Trujillo (2026).

---

Desarrollado para el **TIF Nivel III — UCV Trujillo (2026)** · **ODS 4 · Meta 4.3**.