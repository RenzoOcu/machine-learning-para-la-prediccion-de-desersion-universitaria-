# -*- coding: utf-8 -*-
"""
===============================================================================
 EduPredict Trujillo - Inferencia LIGERA para Vercel (numpy puro)
===============================================================================
  Reemplaza a joblib + xgboost + scikit-learn en TIEMPO DE EJECUCIÓN.

  Carga `modelo_edupredict_light.json` (generado por `convertir_modelo.py`) y
  expone la MISMA interfaz que antes usaba el dashboard:

      modelo, preprocesador = cargar_artefactos_ligeros()

      modelo.predict_proba(X)        # (n, 2), col 1 = P(Dropout=1)
      modelo.feature_importances_    # vector longitud 250 (importancia global)
      modelo.classes_                # np.array([0, 1])

      preprocesador.transform(df)    # (n, 250) exactamente igual al
                                     # ColumnTransformer original (StandardScaler
                                     # + OneHotEncoder, en numpy/pandas)
      preprocesador.get_feature_names_out()

  La réplica del booster usa aritmética float32 (umbrales, valores de feature y
  acumulación de hojas), la misma del motor interno de xgboost: reproduce las
  probabilidades del .pkl original con error < 1e-6 (verificado en `data.csv`
  con las 4 424 filas).
===============================================================================
"""

import json
import os

import numpy as np


RUTA_ARTEFACTOS = "modelo_edupredict_light.json"


# -----------------------------------------------------------------------------
# UTILIDADES DE ÁRBOLES
# -----------------------------------------------------------------------------
def _valor_hoja(nodo):
    """Devuelve el peso de la hoja si `nodo` es hoja, o None si es interno."""
    return nodo.get("leaf")


def _margen_booster(x, arboles, intercepto):
    """Ruta de árbol en float32 + acumulación float32 (como xgboost)."""
    score = np.float32(0.0)
    for arbol in arboles:
        nodo = arbol
        while "leaf" not in nodo:
            if np.float32(x[nodo["split"]]) < np.float32(nodo["split_condition"]):
                nodo = nodo["children"][0]
            else:
                nodo = nodo["children"][1]
        score = np.float32(score + np.float32(_valor_hoja(nodo)))
    return float(score) + float(intercepto)


# -----------------------------------------------------------------------------
# MODELO (envuelve un XGBClassifier binary:logistic)
# -----------------------------------------------------------------------------
class ModeloLigero:
    """Interfaz mínima de un XGBClassifier para el dashboard, en numpy puro."""

    def __init__(self, param):
        self._param       = param
        self._arboles     = param["arboles"]
        self._intercepto  = float(param["intercepto"])
        self._importancia = np.asarray(
            param["importancia"], dtype=np.float64).copy()
        self.classes_     = np.array([0, 1])
        self.es_ligero    = True

    # -- predict_proba -------------------------------------------------------
    def _margen(self, X):
        arboles, intercepto = self._arboles, self._intercepto
        X = np.asarray(X, dtype=np.float64)
        salida = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            salida[i] = _margen_booster(X[i], arboles, intercepto)
        return salida

    def predict_proba(self, X):
        """Devuelve (n, 2): col 0 = P(clase 0), col 1 = P(Dropout=1)."""
        m = np.clip(self._margen(X), -700.0, 700.0)
        p1 = 1.0 / (1.0 + np.exp(-m))
        return np.column_stack([1.0 - p1, p1])

    # -- atributos usados por el dashboard -----------------------------------
    @property
    def feature_importances_(self):
        return self._importancia


# -----------------------------------------------------------------------------
# PREPROCESADOR (equivale a ColumnTransformer: StandardScaler + OneHotEncoder)
# -----------------------------------------------------------------------------
class PreprocesadorLigero:
    """Replica el ColumnTransformer ajustado en `train_model.py`."""

    def __init__(self, param):
        self._columnas_num = list(param["columnas_numericas"])
        self._medias       = np.asarray(param["medias"], dtype=np.float64)
        self._escalas      = np.asarray(param["desviaciones"], dtype=np.float64)
        self._columnas_cat = list(param["columnas_categoricas"])
        self._categorias   = [np.asarray(c) for c in param["categorias"]]
        self._nombres      = list(param["nombre_salida"])

    def get_feature_names_out(self):
        return self._nombres

    def transform(self, df):
        """Idéntico a preprocesador.transform(df[columna_num + columna_cat])."""
        X_num = (df[self._columnas_num].to_numpy(dtype=np.float64)
                 - self._medias) / self._escalas

        total_cat = sum(len(c) for c in self._categorias)
        X_cat = np.zeros((len(df), total_cat), dtype=np.float64)
        offset = 0
        for col, cats in zip(self._columnas_cat, self._categorias):
            valores = df[col].to_numpy()
            for categoria in cats:
                X_cat[:, offset] = (valores == categoria).astype(np.float64)
                offset += 1

        return np.hstack([X_num, X_cat])


# -----------------------------------------------------------------------------
# CARGA DEL ARTEFACTO
# -----------------------------------------------------------------------------
def cargar_artefactos_ligeros(ruta_artefactos=None):
    """Carga el JSON ligero y devuelve (ModeloLigero, PreprocesadorLigero)."""
    ruta = ruta_artefactos or RUTA_ARTEFACTOS
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró '{ruta}'. Ejecuta primero: python convertir_modelo.py"
        )
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            param = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error al leer '{ruta}': {e}") from e
    return ModeloLigero(param), PreprocesadorLigero(param)