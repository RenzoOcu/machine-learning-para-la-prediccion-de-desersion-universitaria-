# -*- coding: utf-8 -*-
"""
===============================================================================
 EduPredict Trujillo - Conversión del modelo a formato LIGERO para Vercel
===============================================================================
  Convierte los artefactos pesados de `train_model.py` (.pkl de XGBoost +
  ColumnTransformer de scikit-learn) en UN solo JSON autocontenido que el
  dashboard puede servir con **numpy puro** (inferencia.py: sin scipy, sin
  xgboost y sin scikit-learn en tiempo de ejecución).

  ¿POR QUÉ?
  Vercel limita el bundle de las funciones Python a 500 MB. El .pkl original
  obliga a instalar xgboost (~150 MB) + scipy (~250 MB) + scikit-learn
  (~120 MB), superando el límite. Con este formato ligero esas librerías se
  mueven a `requirements-train.txt` (solo para retrenar) y el bundle del
  despliegue baja de ~880 MB a ~400 MB.

  ¿QUÉ HACE VERIFICACIÓN NUMÉRICA REAL?
  Tras exportar, recalcula la probabilidad con el pipeline "ligero" para TODAS
  las filas de `data.csv` y la compara con la del modelo XGBoost original. Si
  la diferencia máxima supera 1e-6, el script falla (no se escribe el JSON).
  La réplica usa aritmética float32 (igual que el motor interno de xgboost), lo
  que reproduce las predicciones con un error < 1e-6 en probabilidad.

  USO (una vez por cada vez que se retrene el modelo):
      pip install -r requirements-train.txt
      python convertir_modelo.py

  REQUISITOS (locales, NO van a Vercel):
      joblib, xgboost, scikit-learn, numpy, pandas  -> requirements-train.txt
===============================================================================
"""

import json
import sys
import traceback

import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 0. CONFIGURACIÓN
# -----------------------------------------------------------------------------
SEPARADOR_CSV        = ";"
RUTA_CSV             = "data.csv"
RUTA_MODELO          = "modelo_edupredict.pkl"
RUTA_PREPROCESADOR   = "preprocessor_edupredict.pkl"
ARCHIVO_SALIDA       = "modelo_edupredict_light.json"
TOLERANCIA_MAX       = 1e-6      # Diferencia máxima permitida en probabilidad.

NOMBRE_NUM  = "num"     # Etapa del ColumnTransformer: StandardScaler
NOMBRE_CAT  = "cat"     # Etapa del ColumnTransformer: OneHotEncoder

# Forzamos UTF-8 en consola de Windows (evita errores con tildes)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# -----------------------------------------------------------------------------
# 1. UTILIDADES DE ÁRBOLES XGBOOST
# -----------------------------------------------------------------------------
def _indice_split(split):
    """Convierte 'f12' / '12' / 12 (referencia de feature) en int."""
    if isinstance(split, str) and split.startswith("f"):
        return int(split[1:])
    return int(split)


def _parsear_arbol(texto_json):
    """Convierte un string JSON de xgboost (get_dump) en un dict Python."""
    arbol = json.loads(texto_json)
    if "split" in arbol:
        arbol["split"] = _indice_split(arbol["split"])
    for hijo in arbol.get("children", []):
        _convertir_recursivo(hijo)
    return arbol


def _convertir_recursivo(nodo):
    if "split" in nodo:
        nodo["split"] = _indice_split(nodo["split"])
    nodo.pop("cover", None)          # estadística de entrenamiento innecesaria
    for hijo in nodo.get("children", []):
        _convertir_recursivo(hijo)


# -----------------------------------------------------------------------------
# 2. PREPROCESAMIENTO "LIGERO" (equivalente a StandardScaler + OneHotEncoder)
# -----------------------------------------------------------------------------
def transformar_ligero(df, param):
    """Replica exactamente el ColumnTransformer ajustado, en numpy puro."""
    columnas_num = param["columnas_numericas"]
    medias  = np.asarray(param["medias"], dtype=np.float64)
    escalas = np.asarray(param["desviaciones"], dtype=np.float64)

    X_num = (df[columnas_num].to_numpy(dtype=np.float64) - medias) / escalas

    columnas_cat = param["columnas_categoricas"]
    categorias   = [np.asarray(c) for c in param["categorias"]]
    total_cat = sum(len(c) for c in categorias)
    X_cat = np.zeros((len(df), total_cat), dtype=np.float64)
    offset = 0
    for col, cats in zip(columnas_cat, categorias):
        valores = df[col].to_numpy()
        for categoria in cats:
            X_cat[:, offset] = (valores == categoria).astype(np.float64)
            offset += 1

    return np.hstack([X_num, X_cat])


def margin_ligero(X, param):
    """Margin (score bruto) del booster XGBoost en numpy puro.

    Replica el motor interno de xgboost lo más fielmente posible:
      * umbrales y valores de feature se comparan en float32 (el árbol usa
        float32; los decimales del dump hacen round-trip exacto),
      * los pesos de hoja se acumulan en float32 (como hace predict()).
    El intercepto (logit del base_score) se suma después, resultando la misma
    precisión (~1e-6) que la propia predicción de xgboost.
    """
    arboles    = param["arboles"]
    intercepto = param["intercepto"]
    salida = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        x      = X[i]
        score  = np.float32(0.0)
        for arbol in arboles:
            nodo = arbol
            while "leaf" not in nodo:
                if np.float32(x[nodo["split"]]) < np.float32(nodo["split_condition"]):
                    nodo = nodo["children"][0]
                else:
                    nodo = nodo["children"][1]
            score = np.float32(score + np.float32(nodo["leaf"]))
        salida[i] = float(score) + intercepto
    return salida


def proba_ligera(X, param):
    """predict_proba()[:, 1] exacto (binary:logistic + sigmoid)."""
    m = margin_ligero(X, param)
    m = np.clip(m, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-m))


def importancia_ligera(param):
    """Importancia XGBoost por defecto, igual a `feature_importances_` del .pkl.

    En xgboost 3.x el default es tipo *gain* pero en su variante "media":
    para cada feature, (suma del 'gain' de los nodos donde se divide) entre
    (número de esos nodos), normalizada a suma 1. La réplica coincide con la
    original salvo redondeo float32/XGBoost interno (ordenes visibles iguales).
    """
    n_features = len(param["columnas_numericas"]) + sum(
        len(c) for c in param["categorias"])
    total = np.zeros(n_features, dtype=np.float64)
    conteo = np.zeros(n_features, dtype=np.int64)
    for arbol in param["arboles"]:
        pila = [arbol]
        while pila:
            actual = pila.pop()
            if "leaf" not in actual:
                i = int(actual["split"])
                total[i]   += float(actual.get("gain", 0.0) or 0.0)
                conteo[i]  += 1
                pila += actual["children"]
    media = np.divide(total, conteo,
                      out=np.zeros(n_features), where=conteo > 0)
    suma = media.sum()
    if suma > 0:
        media = media / suma
    return media.tolist()


# -----------------------------------------------------------------------------
# 3. CONVERSIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  EduPredict Trujillo - Conversión a formato ligero")
    print("=" * 72)

    # 3.1 Carga de los artefactos originales
    for ruta in (RUTA_MODELO, RUTA_PREPROCESADOR):
        if not __import__("os").path.exists(ruta):
            print(f"[ERROR] No se encontró '{ruta}'. Ejecuta primero:")
            print("        python train_model.py")
            raise SystemExit(1)

    modelo = joblib.load(RUTA_MODELO)
    prepro = joblib.load(RUTA_PREPROCESADOR)
    print(f"[OK] Cargados: {RUTA_MODELO} y {RUTA_PREPROCESADOR}")

    # 3.2 Parámetros del preprocesador (StandardScaler + OneHotEncoder)
    scaler  = prepro.named_transformers_[NOMBRE_NUM]
    encoder = prepro.named_transformers_[NOMBRE_CAT]

    columnas_num = list(scaler.feature_names_in_)
    columnas_cat = list(encoder.feature_names_in_)
    categorias   = [c.tolist() for c in encoder.categories_]

    param = {
        "columnas_numericas":   columnas_num,
        "medias":               scaler.mean_.tolist(),
        "desviaciones":         scaler.scale_.tolist(),
        "columnas_categoricas": columnas_cat,
        "categorias":           categorias,
        "nombre_salida":        list(prepro.get_feature_names_out()),
        "arboles":              None,   # se rellena más abajo
        "intercepto":           0.0,    # se rellena más abajo
        "importancia":          None,   # se rellena más abajo
    }
    print(f"[OK] Preprocesador: {len(columnas_num)} numéricas + "
          f"{len(columnas_cat)} categóricas "
          f"({sum(len(c) for c in categorias)} columnas one-hot).")

    # 3.3 Booster XGBoost: base_score + árboles
    booster = modelo.get_booster()
    config  = json.loads(booster.save_config())
    base_score_bruto = config["learner"]["learner_model_param"]["base_score"]
    base_score = float(str(base_score_bruto).strip("[] "))
    intercepto = float(np.log(base_score / (1.0 - base_score)))

    arboles = [_parsear_arbol(t)
               for t in booster.get_dump(dump_format="json", with_stats=True)]
    param["arboles"]     = arboles
    param["intercepto"]  = intercepto
    param["importancia"] = importancia_ligera(param)
    print(f"[OK] Booster: {len(arboles)} árboles | "
          f"base_score={base_score:.6f} | intercepto={intercepto:.6f} | "
          f"features={len(param['importancia'])}")

    # 3.4 VERIFICACIÓN NUMÉRICA sobre TODAS las filas del dataset
    df = pd.read_csv(RUTA_CSV, sep=SEPARADOR_CSV)
    df.columns = [c.strip() for c in df.columns]
    columnas_modelo = columnas_num + columnas_cat
    X_ref = prepro.transform(df[columnas_modelo])
    p_ref = modelo.predict_proba(X_ref)[:, 1]

    X_lig  = transformar_ligero(df, param)
    p_lig  = proba_ligera(X_lig, param)
    dif    = float(np.abs(p_lig - p_ref).max())

    print(f"\n[+] Verificación sobre {len(df)} filas:")
    print(f"    Probabilidad original (xgboost) : media={p_ref.mean():.6f}")
    print(f"    Probabilidad ligera (numpy)     : media={p_lig.mean():.6f}")
    print(f"    Diferencia máxima               : {dif:.3e} "
          f"(tol: {TOLERANCIA_MAX:.0e})")
    if dif > TOLERANCIA_MAX:
        print("[ERROR] Los resultados NO coinciden. No se escribe el JSON.")
        raise SystemExit(1)
    print("[OK] Verificación superada: el modelo ligero reproduce el original.")

    # 3.5 Guardado del artefacto único
    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        json.dump(param, f, ensure_ascii=False, separators=(",", ":"))
    tam_mb = __import__("os").path.getsize(ARCHIVO_SALIDA) / (1024 * 1024)
    print(f"[OK] Guardado -> {ARCHIVO_SALIDA} ({tam_mb:.2f} MB)")

    print("\n" + "=" * 72)
    print("  ¡CONVERSIÓN COMPLETADA!")
    print("  El dashboard ahora carga el modelo sin xgboost/scipy/scikit-learn.")
    print("  Recuerda subir 'modelo_edupredict_light.json' al deploy.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        raise SystemExit(1)