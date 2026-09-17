# -*- coding: utf-8 -*-
"""
===============================================================================
 EduPredict Trujillo - Entrenamiento del modelo predictivo de deserción
===============================================================================
  TIF Nivel III - Universidad César Vallejo (Trujillo, 2026).
  Sistema predictivo de deserción universitaria basado en Machine Learning.

  OBJETIVO DE DESARROLLO SOSTENIBLE:
    ODS 4 "Educación de calidad" -> Meta 4.3: acceso igualitario a la
    educación superior (técnica, profesional y universitaria).

  DATASET (IMPORTANTE - NO NEGOCIABLE):
    Este script SOLO funciona con el dataset
    "Predict students' dropout and academic success" del UCI Machine Learning
    Repository (4.424 estudiantes | 36 variables predictoras + Target).

    Descarga el archivo `data.csv` (separador ';') desde:
    https://uci-ics-mlr-prod.aws.uci.edu/dataset/697/predict+students+dropout+and+academic+success
    y colócalo en la misma carpeta que este script.

  FLUJO DEL ENTRENAMIENTO:
    1. Carga `data.csv` con sep=';'.
    2. Binariza 'Target' -> Dropout = 1 (clase minoritaria), Graduate/Enrolled = 0.
    3. Divide en train/test (80/20) con estratificación (stratify=y) y random_state=42.
    4. ColumnTransformer: StandardScaler (numéricas) + OneHotEncoder (categóricas).
    5. Pipeline de imbalanced-learn: Preprocesador -> SMOTE (SOLO train) -> XGBoost.
       Al insertar SMOTE DENTRO del pipeline, el balanceo se aplica únicamente a los
       datos de entrenamiento y se evita el data leakage sobre el conjunto de test.
    6. Evaluación sobre test (sin SMOTE): Accuracy, Recall, F1-Score y AUC-ROC.
    7. Guarda con joblib el modelo y el preprocesador ya ajustados.

  ARCHIVOS GENERADOS:
    - modelo_edupredict.pkl        -> Clasificador XGBoost entrenado.
    - preprocessor_edupredict.pkl  -> ColumnTransformer ajustado (escalador + OHE).
    - metricas_edupredict.json     -> Métricas calculadas (mostradas en el dashboard).
===============================================================================
"""

import json
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             roc_auc_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# 0. CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------
SEPARADOR_CSV      = ";"          # Separador real del dataset UCI.
RUTA_CSV           = "data.csv"   # Dataset UCI (debe estar junto al script).
COLUMNA_TARGET     = "Target"     # Variable objetivo original (3 clases).
RANDOM_STATE       = 42           # Reproducibilidad total del experimento.
TEST_SIZE          = 0.2          # 80% entrenamiento / 20% prueba.

# Nombres de los artefactos guardados
ARCHIVO_MODELO        = "modelo_edupredict.pkl"
ARCHIVO_PREPROCESADOR = "preprocessor_edupredict.pkl"
ARCHIVO_METRICAS      = "metricas_edupredict.json"

# Nombre literal de las etapas del pipeline (para extraer cada pieza)
NOMBRE_PREPROC   = "preprocessor"
NOMBRE_SMOTE     = "smote"
NOMBRE_CLASIF    = "classifier"

# Forzamos salida UTF-8 en consola de Windows (evita errores con tilde)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -----------------------------------------------------------------------------
# 1. DEFINICIÓN DE COLUMNAS REALES DEL DATASET UCI
# -----------------------------------------------------------------------------
# NOTA DE INGENIERÍA:
# Aunque "Previous qualification (grade)" suene a categoría, en el dataset UCI
# REAL es una nota numérica continua (0 - 200). Aplicarle One-Hot Encoding
# generaría cientos de columnas dummy y destruiría su naturaleza ordinal.
# Por eso se escala con StandardScaler. El resto de columnas que aparecen aquí
# sí son categóricas discretas en el dataset original.

COLUMNAS_NUMERICAS = [
    "Previous qualification (grade)",          # Nota previa (0-200, continua)
    "Admission grade",                         # Nota de acceso a la universidad (0-200)
    "Age at enrollment",                       # Edad al matricularse (17-70)
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",                       # Tasa de desempleo (%)
    "Inflation rate",                          # Tasa de inflación (%)
    "GDP",                                     # Producto Interno Bruto
]

COLUMNAS_CATEGORICAS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Course",
]

# Total de columnas predictoras (36) = 18 numéricas + 18 categóricas
COLUMNAS_PREDICTORAS = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS


# -----------------------------------------------------------------------------
# 2. FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    """
    Orquesta todo el flujo: carga -> transformación -> balanceo -> modelo ->
    evaluación -> guardado de artefactos del sistema EduPredict Trujillo.
    """
    print("=" * 72)
    print("  EduPredict Trujillo - Entrenamiento del modelo predictivo")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 2.1 CARGA DEL DATASET UCI (separador ';')
    # -------------------------------------------------------------------------
    try:
        df = pd.read_csv(RUTA_CSV, sep=SEPARADOR_CSV)
        # El CSV original puede traer espacios/tabs alrededor de algunos
        # nombres de columna; se limpian para evitar errores de coincidencia.
        df.columns = [c.strip() for c in df.columns]
        print(f"[OK] Dataset cargado: {RUTA_CSV}")
        print(f"     Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    except FileNotFoundError:
        print("[ERROR] No se encontró 'data.csv'.")
        print("        Descárgalo desde el UCI ML Repository:")
        print("        https://uci-ics-mlr-prod.aws.uci.edu/dataset/697/predict+students+dropout+and+academic+success")
        print("        y colócalo en la misma carpeta que este script.")
        raise SystemExit(1)
    except Exception as e:
        print(f"[ERROR] No se pudo leer el dataset: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    # -------------------------------------------------------------------------
    # 2.2 VALIDACIÓN DE COLUMNAS (estructura EXACTA del dataset UCI)
    # -------------------------------------------------------------------------
    faltantes = [c for c in COLUMNAS_PREDICTORAS + [COLUMNA_TARGET]
                 if c not in df.columns]
    if faltantes:
        print("[ERROR] El dataset no tiene la estructura esperada del UCI.")
        print("        Columnas ausentes:", faltantes)
        raise SystemExit(1)
    print("[OK] Estructura de columnas validada (37 columnas reales del UCI).")

    # -------------------------------------------------------------------------
    # 2.3 BINARIZACIÓN DE LA VARIABLE OBJETIVO
    #   Dropout  = 1 (CLASE MINORITARIA -> la que queremos detectar)
    #   Graduate / Enrolled = 0
    # -------------------------------------------------------------------------
    df[COLUMNA_TARGET] = df[COLUMNA_TARGET].astype(str).str.strip().map(
        {"Dropout": 1, "Graduate": 0, "Enrolled": 0}
    )
    # La distribución original de clases en el dataset UCI es aproximada:
    #   Dropout ~32%, Graduate ~49%, Enrolled ~19%.
    print("\n[+] Distribución original de clases (Target binarizado):")
    print(df[COLUMNA_TARGET].value_counts().to_string())
    print(f"    % de deserción (Dropout=1): "
          f"{(df[COLUMNA_TARGET].mean()*100):.2f}%")

    # -------------------------------------------------------------------------
    # 2.4 SEPARACIÓN DE VARIABLES Y DIVISIÓN TRAIN/TEST
    # -------------------------------------------------------------------------
    X = df[COLUMNAS_PREDICTORAS]
    y = df[COLUMNA_TARGET]

    # División 80/20 estratificada para mantener la proporción de desertores
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\n[OK] División train/test: "
          f"train={len(X_train)} filas | test={len(X_test)} filas")

    # -------------------------------------------------------------------------
    # 2.5 COLUMNTRANSFORMER: StandardScaler + OneHotEncoder
    # -------------------------------------------------------------------------
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLUMNAS_NUMERICAS),   # Escalado numérico
            ("cat", OneHotEncoder(handle_unknown="ignore",    # Tolerancia a
                                  sparse_output=False),       # categorías nuevas
             COLUMNAS_CATEGORICAS),
        ],
        # sparse_output=False: OneHotEncoder devuelve numpy denso, mucho más
        # eficiente para XGBoost y SMOTE (que internamente usa KNN).
    )

    # -------------------------------------------------------------------------
    # 2.6 PIPELINE IMBALANCED-LEARN: Preprocesador -> SMOTE -> XGBoost
    # -------------------------------------------------------------------------
    # Usamos imblearn.pipeline.Pipeline (NO el de sklearn) para que SMOTE sea
    # una etapa válida del pipeline. SMOTE solo se ejecuta durante .fit() sobre
    # el conjunto de entrenamiento; nunca toca el test en .predict().
    pipeline = ImbPipeline([
        (NOMBRE_PREPROC, preprocesador),
        (NOMBRE_SMOTE, SMOTE(random_state=RANDOM_STATE)),
        (NOMBRE_CLASIF, XGBClassifier(
            n_estimators=200,          # 200 árboles (gradient boosting)
            learning_rate=0.1,         # 1 peso de paso por árbol
            max_depth=6,               # Profundidad de los árboles
            random_state=RANDOM_STATE,
            eval_metric="logloss",     # Métrica interna del booster
            n_jobs=-1,                 # Usar todos los núcleos de CPU
            verbosity=0,
        )),
    ])

    # Entrenamiento: el pipeline hace transforms -> SMOTE(train) -> XGBoost.fit
    print("\n[+] Entrenando pipeline (Preprocesador + SMOTE + XGBoost)...")
    pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 2.7 EVALUACIÓN SOBRE TEST (el test NO recibe SMOTE ni es.SMU)
    # -------------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]   # Probabilidad de clase 1

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)  # Precisión sobre Dropout
    recall   = recall_score(y_test, y_pred, pos_label=1)   # Recall sobre Dropout
    f1       = f1_score(y_test, y_pred, pos_label=1)
    auc_roc  = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 72)
    print("  MÉTRICAS DE EVALUACIÓN (dataset de prueba, sin SMOTE)")
    print("=" * 72)
    print(f"  Accuracy           : {accuracy:.4f}")
    print(f"  Precision (Dropout): {precision:.4f}")
    print(f"  Recall (Dropout)   : {recall:.4f}   <- capacidad de detectar desertores")
    print(f"  F1-Score (Dropout) : {f1:.4f}")
    print(f"  AUC-ROC            : {auc_roc:.4f}")
    print("=" * 72)

    print("\n[+] Reporte de clasificación detallado:")
    print(classification_report(y_test, y_pred, target_names=[
        "Graduado/Enrolled (0)", "Dropout (1)"
    ]))

    print("\n[+] Matriz de confusión:")
    print(np.array2string(
        cm,
        separator="  "
    ))

    # -------------------------------------------------------------------------
    # 2.71 VALIDACIÓN DEL ORDEN DEL PIPELINE (anti data leakage)
    # -------------------------------------------------------------------------
    # Estructura garantizada por imblearn.pipeline.Pipeline:
    #   1) ColumnTransformer.fit_transform sobre el TRAIN (escalar + one-hot)
    #   2) SMOTE.fit_resample SOLO sobre el TRAIN
    #   3) XGBClassifier.fit sobre el TRAIN ya balanceado
    #   En .predict() el pipeline NO tiene paso SMOTE: el TEST nunca es
    #   balanceado ni visto por SMOTE.
    print("\n[OK] Verificación anti data leakage:")
    print("     1. train_test_split estratificado (80/20) -> random_state=42")
    print("     2. ColumnTransformer ajustado solo en TRAIN")
    print("     3. SMOTE DENTRO del pipeline: actúa SOLO sobre TRAIN")
    print("     4. XGBoost entrenado sobre TRAIN balanceado")
    print("     5. Métricas evaluadas sobre TEST original (sin SMOTE)")

    # -------------------------------------------------------------------------
    # 2.8 GUARDADO DE ARTEFACTOS CON JOBLIB
    # -------------------------------------------------------------------------
    # Extraemos del pipeline ajustado:
    #   - el preprocesador (para transformar las entradas del dashboard)
    #   - el clasificador (ya entrenado sobre los datos balanceados)
    modelo = pipeline.named_steps[NOMBRE_CLASIF]
    preproc_ajustado = pipeline.named_steps[NOMBRE_PREPROC]

    joblib.dump(modelo, ARCHIVO_MODELO)
    joblib.dump(preproc_ajustado, ARCHIVO_PREPROCESADOR)
    print(f"\n[OK] Modelo guardado        -> {ARCHIVO_MODELO}")
    print(f"[OK] Preprocesador guardado  -> {ARCHIVO_PREPROCESADOR}")

    # Métricas exportadas a JSON (el dashboard las muestra sin reevaluar)
    metricas = {
        "accuracy": float(accuracy),
        "precision_dropout": float(precision),
        "recall_dropout": float(recall),
        "f1_dropout": float(f1),
        "auc_roc": float(auc_roc),
        "confusion_matrix": cm.tolist(),
        "clases_modelo": [int(c) for c in pipeline.classes_],
        "distribucion_original": {
            str(k): int(v) for k, v in df[COLUMNA_TARGET].value_counts().items()
        },
        "distribucion_binarizada": {
            "Dropout(1)": int((df[COLUMNA_TARGET] == 1).sum()),
            "Graduado/Enrolled(0)": int((df[COLUMNA_TARGET] == 0).sum()),
        },
        "interpretacion_clases": (
            "El modelo se entrenó binarizando Target: Dropout=1 (clase de "
            "interés) frente a Graduate/Enrolled=0. El 'riesgo de deserción' "
            "mostrado en el dashboard es P(clase=1) de predict_proba()."
        ),
        "umbrales_riesgo": {
            "BAJO": "0-30%", "MEDIO": "30-60%", "ALTO": ">=60%",
            "nota": "Umbrales definidos por la APLICACIÓN para visualización; "
                    "no son categorías aprendidas por el modelo.",
        },
    }
    with open(ARCHIVO_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"[OK] Métricas guardadas     -> {ARCHIVO_METRICAS}")

    print("\n" + "=" * 72)
    print("  ¡ENTRENAMIENTO COMPLETADO!")
    print("  Ejecuta ahora el dashboard:  streamlit run app.py")
    print("=" * 72)


# -----------------------------------------------------------------------------
# 3. PUNTO DE ENTRADA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()