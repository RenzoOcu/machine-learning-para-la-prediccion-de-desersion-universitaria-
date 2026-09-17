# -*- coding: utf-8 -*-
"""
===============================================================================
 EduPredict Trujillo - Dashboard Web Interactivo (Streamlit)
===============================================================================
  Sistema predictivo de deserción universitaria (TIF Nivel III, UCV Trujillo,
  2026).

  ADAPTACIÓN / PROPUESTA DE APLICACIÓN AL CONTEXTO DE TRUJILLO (PERÚ)
  -------------------------------------------------------------------
  El modelo de Machine Learning es el mismo entrenado con el dataset oficial
  "Predict students' dropout and academic success" del UCI ML Repository.
  El dataset NO fue recolectado en Perú: fue recolectado en Portugal. Por lo
  tanto, este sistema es una *adaptación/propuesta de aplicación* del modelo
  al contexto universitario de Trujillo y NO afirma representar
  estadísticamente a todos los estudiantes peruanos.

  VERSIÓN SIMPLE / ADAPTADA A PERÚ:
  - El formulario pregunta SOLO las variables que un tutor puede obtener con
    facilidad (edad, sexo, carrera, turno, nota de admisión 0-20, resultados de
    los 2 primeros ciclos en escala 0-20, nivel de estudios de los padres,
    situación económica según RMV S/1,130, beca y pensiones).
  - Las variables restantes que el modelo necesita se COMPLETAN con valores
    documentados (modas del dataset) y se muestran al usuario en "Ver detalle
    técnico" / "Cómo se completó el formulario". NINGÚN valor se rellena a
    escondidas: el detalle se muestra siempre en el resultado.

  TRANSFORMACIONES DOCUMENTADAS (interfaz peruana -> formato del modelo):
    1) Nota de admisión (0-20 peruana) -> Admission grade (escala ×10). La
       misma nota se usa como aproximación de Previous qualification (grade)
       (correlación 0.58 en el dataset; se documenta en el resultado).
    2) Nota promedio de unidades curriculares (0-20): coincide con la escala
       original del dataset (0-20), NO se transforma.
    3) Situación económica familiar (RMV S/1,130): parámetro DE CONTEXTO de la
       aplicación que se traduce a 3 variables macro originales del dataset
       (Unemployment rate, Inflation rate, GDP). NO son datos originales.
    4) Cursos evaluados = cursos matriculados (supuesto documentado: todas las
       asignaturas matriculadas se evalúan). Acreditaciones y cursos sin
       evaluación = 0 (lo más frecuente al ingresar).

  Archivos requeridos (generados por `train_model.py`):
    - modelo_edupredict.pkl        (clasificador XGBoost)
    - preprocessor_edupredict.pkl  (ColumnTransformer: StandardScaler + OHE)

  Ejecución:  streamlit run app.py
===============================================================================
"""

import json
import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Plotly en modo claro (tema Material/Google del dashboard).
pio.templates.default = "plotly_white"

import streamlit as st

# SHAP (explicabilidad) es OPCIONAL: si no está instalado se usa un fallback.
try:
    import shap
    SHAP_DISPONIBLE = True
except ImportError:
    SHAP_DISPONIBLE = False

# Forzamos UTF-8 en consola de Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -----------------------------------------------------------------------------
# 0. CONFIGURACIÓN GLOBAL DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict Trujillo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUTA_MODELO        = "modelo_edupredict.pkl"
RUTA_PREPROCESADOR = "preprocessor_edupredict.pkl"
RUTA_METRICAS      = "metricas_edupredict.json"
RUTA_DATASET       = "data.csv"      # Solo para la pestaña de análisis
URL_UCI            = ("https://uci-ics-mlr-prod.aws.uci.edu/dataset/697/"
                      "predict+students+dropout+and+academic+success")

COLOR_AZUL  = "#4285F4"
COLOR_VERDE = "#34a853"
COLOR_AMBAR = "#fbbc04"
COLOR_ROJO  = "#ea4335"
COLOR_TEXTO = "#202124"
COLOR_FONDO = "#f8f9fa"
COLOR_FONDO_SEC = "#f1f3f4"

# CSS global: tema claro estilo Google / Material Design
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Roboto', 'Google Sans', 'Segoe UI', sans-serif;
        }}

        .stApp {{ background-color: {COLOR_FONDO}; }}
        header[data-testid="stHeader"] {{ display: none; }}

        h1, h2, h3, h4 {{ color: {COLOR_TEXTO}; font-weight: 500; }}

        /* Animación de entrada suave */
        @keyframes edubrightFadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .block-container {{
            padding-top: 2rem;
            animation: edubrightFadeIn .45s ease-out both;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            transition: background-color .2s ease, color .2s ease;
        }}

        /* Tarjetas estilo Material */
        .tarjeta {{
            border-radius: 12px; padding: 1rem 1.2rem; color: {COLOR_TEXTO};
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 3px rgba(60,64,67,.12), 0 2px 6px rgba(60,64,67,.08);
            margin-bottom: 0.7rem;
            transition: box-shadow .25s ease, transform .25s ease;
        }}
        .tarjeta:hover {{
            box-shadow: 0 4px 12px rgba(60,64,67,.20);
            transform: translateY(-2px);
        }}
        .tarjeta-azul  {{ border-top: 4px solid {COLOR_AZUL};  }}
        .tarjeta-verde {{ border-top: 4px solid {COLOR_VERDE}; }}
        .tarjeta-roja  {{ border-top: 4px solid {COLOR_ROJO};  }}
        .tarjeta-ambar {{ border-top: 4px solid {COLOR_AMBAR}; }}

        .recomendacion {{
            border-left: 5px solid {COLOR_AZUL}; padding: 0.4rem 1rem;
            border-radius: 8px; background-color: {COLOR_FONDO_SEC};
            color: {COLOR_TEXTO}; margin: 0.6rem 0;
        }}
        .explicacion {{
            font-size: 0.92rem; color: #5f6368; background-color: #ffffff;
            border: 1px solid #e8eaed; border-radius: 10px;
            padding: 0.8rem 1rem; margin: 0.4rem 0 1rem 0;
            line-height: 1.5;
        }}
        .explicacion b {{ color: {COLOR_TEXTO}; }}
        .paso-indicador {{ color: #5f6368; font-size: 0.9rem; margin: 0.2rem 0 0.6rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 1. COLUMNAS REALES DEL DATASET UCI (36 predictoras)
# -----------------------------------------------------------------------------
COLUMNAS_NUMERICAS = [
    "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]

COLUMNAS_CATEGORICAS = [
    "Marital status", "Application mode", "Application order",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
    "Course",
]

COLUMNAS_TOTALES = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS

# -----------------------------------------------------------------------------
# 2. CATEGORÍAS REALES DEL DATASET UCI (codificación original portuguesa)
#    Etiquetas traducidas al español según la documentación oficial del UCI ML
#    Repository. El CÓDIGO numérico es el que alimenta al modelo (integridad
#    de categorías); la etiqueta es solo para mostrar opciones legibles.
# -----------------------------------------------------------------------------
NOMBRES_MODALIDAD = {
    1: "1.ª fase – contingente general",
    2: "Decreto-ley 612/93",
    5: "1.ª fase – contingente especial (isla de Azores)",
    7: "Titulares de otros cursos superiores",
    10: "Decreto-ley 854-B/99",
    15: "Estudiante internacional (grado)",
    16: "1.ª fase – contingente especial (isla de Madeira)",
    17: "2.ª fase – contingente general",
    18: "3.ª fase – contingente general",
    26: "Decreto-ley 533-A/99, punto b2 (plan diferente)",
    27: "Decreto-ley 533-A/99, punto b3 (otra institución)",
    39: "Mayores de 23 años",
    42: "Transferencia",
    43: "Cambio de curso",
    44: "Titulares de diplomas de especialización tecnológica",
    51: "Cambio de institución/curso",
    53: "Titulares de diploma de ciclo corto",
    57: "Cambio de institución/curso (internacional)",
}
NOMBRES_CARRERAS = {
    33: "Producción de biocombustibles",
    171: "Diseño de animación y multimedia",
    8014: "Servicio social (horario vespertino)",
    9003: "Agronomía",
    9070: "Diseño de la comunicación",
    9085: "Enfermería veterinaria",
    9119: "Ingeniería informática",
    9130: "Equinicultura",
    9147: "Administración",
    9238: "Servicio social",
    9254: "Turismo",
    9500: "Enfermería",
    9556: "Higiene bucodental",
    9670: "Gestión de publicidad y marketing",
    9773: "Periodismo y comunicación",
    9853: "Educación básica",
    9991: "Administración (horario vespertino)",
}
NOMBRES_NACIONALIDADES = {
    1: "Portuguesa", 2: "Alemana", 6: "Española", 11: "Italiana",
    13: "Neerlandesa (holandesa)", 14: "Inglesa", 17: "Lituana",
    21: "Angoleña", 22: "Caboverdiana", 24: "Guineana", 25: "Mozambiqueña",
    26: "Santomense", 32: "Turca", 41: "Brasileña", 62: "Rumana",
    100: "Moldava", 101: "Mexicana", 103: "Ucraniana", 105: "Rusa",
    108: "Cubana", 109: "Colombiana",
}
NOMBRES_FORMACION_PREVIA = {
    1: "Educación secundaria",
    2: "Estudios superiores – grado (bachelor)",
    3: "Estudios superiores – título de grado",
    4: "Estudios superiores – maestría",
    5: "Estudios superiores – doctorado",
    6: "Cursó estudios superiores (sin título)",
    9: "12.º año de escolaridad – no completado",
    10: "11.º año de escolaridad – no completado",
    12: "Otros – 11.º año de escolaridad",
    14: "10.º año de escolaridad",
    15: "10.º año de escolaridad – no completado",
    19: "Educación básica 3.er ciclo (9.º/10.º/11.º) o equivalente",
    38: "Educación básica 2.º ciclo (6.º/7.º/8.º) o equivalente",
    39: "Curso de especialización tecnológica",
    40: "Estudios superiores – grado (1.er ciclo)",
    42: "Curso técnico superior profesional",
    43: "Estudios superiores – maestría (2.º ciclo)",
}
BASES_NIVEL_EDUCATIVO = {
    1: "Educación secundaria – 12.º año de escolaridad o equivalencia",
    2: "Estudios superiores – grado (bachelor)",
    3: "Estudios superiores – título de grado",
    4: "Estudios superiores – maestría",
    5: "Estudios superiores – doctorado",
    6: "Cursó estudios superiores (sin título)",
    9: "12.º año de escolaridad – no completado",
    10: "11.º año de escolaridad – no completado",
    11: "7.º año (sistema antiguo)",
    12: "Otros – 11.º año de escolaridad",
    14: "10.º año de escolaridad",
    18: "Curso general de comercio",
    19: "Educación básica 3.er ciclo (9.º/10.º/11.º) o equivalente",
    22: "Curso técnico-profesional",
    26: "7.º año de escolaridad",
    27: "2.º ciclo del curso de secundaria general",
    29: "9.º año de escolaridad – no completado",
    30: "8.º año de escolaridad",
    34: "Desconocido",
    35: "No sabe leer ni escribir",
    36: "Sabe leer sin tener el 4.º año",
    37: "Educación básica 1.er ciclo (4.º/5.º año) o equivalente",
    38: "Educación básica 2.º ciclo (6.º/7.º/8.º año) o equivalente",
    39: "Curso de especialización tecnológica",
    40: "Estudios superiores – grado (1.er ciclo)",
    41: "Curso de estudios superiores especializados",
    42: "Curso técnico superior profesional",
    43: "Estudios superiores – maestría (2.º ciclo)",
    44: "Estudios superiores – doctorado (3.er ciclo)",
}
NOMBRES_NIVEL_MADRE = dict(BASES_NIVEL_EDUCATIVO)
NOMBRES_NIVEL_PADRE = {
    **BASES_NIVEL_EDUCATIVO,
    13: "2.º año del curso complementario de secundaria",
    20: "Curso de secundaria complementaria",
    25: "Curso de secundaria complementaria – no concluido",
    31: "Curso general de administración y comercio",
    33: "Contabilidad y administración complementarias",
}
NOMBRES_OCU_MADRE = {
    0: "Estudiante",
    1: "Representantes del poder legislativo y de órganos ejecutivos, "
       "directores y gerentes",
    2: "Especialistas en actividades intelectuales y científicas",
    3: "Técnicos y profesiones de nivel intermedio",
    4: "Personal administrativo",
    5: "Trabajadores de servicios personales, de protección y seguridad y "
       "vendedores",
    6: "Agricultores y trabajadores cualificados de agricultura, pesca y "
       "silvicultura",
    7: "Trabajadores cualificados de industria, construcción y artesanos",
    8: "Operadores de instalaciones y máquinas y trabajadores de montaje",
    9: "Trabajadores no cualificados",
    10: "Profesiones de las Fuerzas Armadas",
    90: "Otra situación",
    99: "(en blanco)",
    122: "Profesionales de la salud",
    123: "Profesores y docentes",
    125: "Especialistas en tecnologías de la información y comunicación (TIC)",
    131: "Técnicos y profesiones de nivel intermedio de ciencias e ingeniería",
    132: "Técnicos y profesionales de nivel intermedio de la salud",
    134: "Técnicos de nivel intermedio de servicios jurídicos, sociales, "
         "deportivos, culturales y similares",
    141: "Empleados de oficina, secretarios en general y operadores de "
         "procesamiento de datos",
    143: "Operadores de datos, contabilidad, estadística, servicios "
         "financieros y de registro",
    144: "Otro personal de apoyo administrativo",
    151: "Trabajadores de servicios personales",
    152: "Vendedores",
    153: "Cuidadores de personas y similares",
    171: "Trabajadores cualificados de la construcción y similares, excepto "
         "electricistas",
    173: "Trabajadores cualificados de imprenta, fabricación de instrumentos "
         "de precisión, joyeros, artesanos y similares",
    175: "Trabajadores de procesamiento de alimentos, carpintería, confección "
         "y otras industrias y oficios artesanales",
    191: "Trabajadores de limpieza",
    192: "Trabajadores no cualificados de agricultura, producción animal, "
         "pesca y silvicultura",
    193: "Trabajadores no cualificados de industria extractiva, construcción, "
         "manufactura y transporte",
    194: "Ayudantes de preparación de comidas",
}
NOMBRES_OCU_PADRE = {
    **NOMBRES_OCU_MADRE,
    101: "Oficiales de las Fuerzas Armadas",
    102: "Sargentos de las Fuerzas Armadas",
    103: "Otro personal de las Fuerzas Armadas",
    112: "Directores de servicios administrativos y comerciales",
    114: "Directores de hoteles, restauración, comercio y otros servicios",
    121: "Especialistas en ciencias físicas, matemáticas, ingeniería y "
         "técnicas afines",
    124: "Especialistas en finanzas, contabilidad, organización "
         "administrativa y relaciones públicas y comerciales",
    135: "Técnicos de tecnologías de la información y comunicación",
    154: "Personal de servicios de protección y seguridad",
    161: "Agricultores orientados al mercado y trabajadores cualificados de "
         "producción agrícola y animal",
    163: "Agricultores, ganaderos, pescadores, cazadores y recolectores de "
         "subsistencia",
    172: "Trabajadores cualificados de metalurgia, metalmecánica y similares",
    174: "Trabajadores cualificados de electricidad y electrónica",
    181: "Operadores de instalaciones fijas y de máquinas",
    182: "Trabajadores de montaje (ensamblaje)",
    183: "Conductores de vehículos y operadores de equipos móviles",
    195: "Vendedores ambulantes (excepto alimentos) y proveedores de "
         "servicios en la calle",
}

CODES_APPLICATION_MODE = sorted(NOMBRES_MODALIDAD)
CODES_NACIONALITY       = sorted(NOMBRES_NACIONALIDADES)
CODES_PREV_QUAL         = sorted(NOMBRES_FORMACION_PREVIA)
CODES_QUAL_MOTHER       = sorted(NOMBRES_NIVEL_MADRE)
CODES_QUAL_FATHER       = sorted(NOMBRES_NIVEL_PADRE)
CODES_OCU_MOTHER        = sorted(NOMBRES_OCU_MADRE)
CODES_OCU_FATHER        = sorted(NOMBRES_OCU_PADRE)
CODES_COURSE            = sorted(NOMBRES_CARRERAS)

# Código -> diccionario de etiquetas, por columna del modelo (para tablas).
MAPA_CODIGOS = {
    "Application mode": NOMBRES_MODALIDAD,
    "Nacionality": NOMBRES_NACIONALIDADES,
    "Previous qualification": NOMBRES_FORMACION_PREVIA,
    "Mother's qualification": NOMBRES_NIVEL_MADRE,
    "Father's qualification": NOMBRES_NIVEL_PADRE,
    "Mother's occupation": NOMBRES_OCU_MADRE,
    "Father's occupation": NOMBRES_OCU_PADRE,
    "Course": NOMBRES_CARRERAS,
}


def _codigo(valor, default):
    """Extrae el código numérico de un valor de widget (label, codigo)."""
    if isinstance(valor, tuple):
        return valor[1]
    return default


def _nombre(dicc, codigo):
    """Resultado: 'Nombre (código N)'. Nunca inventa etiquetas."""
    nombre = dicc.get(int(codigo))
    if nombre is None:
        return f"Código {codigo} (no catalogado)"
    return f"{nombre} (código {codigo})"


def _nombre_categorias(fila):
    """Devuelve la fila con etiquetas legibles para columnas categóricas."""
    out = {}
    for col, valor in fila.items():
        dicc = MAPA_CODIGOS.get(col)
        if dicc is not None:
            out[col] = _nombre(dicc, valor)
        else:
            out[col] = valor
    return out


# -----------------------------------------------------------------------------
# 3-A. VERSIÓN PERUANA SIMPLIFICADA
#      Carreras, formación previa y nivel educativo de los padres en términos
#      de Perú, mapeados a los CÓDIGOS reales del dataset UCI (integridad).
# -----------------------------------------------------------------------------
NOMBRES_CARRERAS_PERU = {
    "Ingeniería de Sistemas e Informática": 9119,
    "Administración de Empresas": 9147,
    "Enfermería": 9500,
    "Turismo y Hotelería": 9254,
    "Agronomía / Ingeniería Agrónoma": 9003,
    "Comunicación y Periodismo": 9773,
    "Educación (Docencia)": 9853,
    "Trabajo Social": 9238,
    "Diseño Gráfico y Comunicación Visual": 9070,
    "Marketing y Publicidad": 9670,
    "Medicina Veterinaria": 9085,
    "Animación y Multimedia": 171,
    "Dentistería / Higiene Bucal": 9556,
    "Agroindustria y Biocombustibles": 33,
}
# Algunas carreras tienen versión vespertina en el dataset (código distinto).
CARRERA_VESPERTINO = {9238: 8014, 9147: 9991}
NOMBRE_CARRERA_POR_CODIGO = {v: k for k, v in NOMBRES_CARRERAS_PERU.items()}
NOMBRE_CARRERA_POR_CODIGO.update({
    8014: "Trabajo Social (nocturno)",
    9991: "Administración (nocturno)",
})
# La tabla técnica muestra el nombre peruano de la carrera (códigos UCI reales
# asignados a las carreras equivalentes peruanas).
MAPA_CODIGOS["Course"] = NOMBRE_CARRERA_POR_CODIGO

# Nivel de estudios de los padres en términos peruanos -> código UCI real.
NIVELES_EDU_PADRE_PERU = {
    "No estudió": 35,
    "Primaria": 37,
    "Secundaria incompleta": 19,
    "Secundaria completa": 1,
    "Instituto / carrera técnica": 42,
    "Universidad incompleta": 6,
    "Universidad completa": 3,
    "Maestría / Doctorado": 43,
    "No sabe / no responde": 34,
}

# Ocupación de los padres ESTIMADA según su nivel de estudios (moda del
# dataset por nivel educativo). Solo se usa para completar el formulario.
OCU_MADRE_POR_NIVEL = {
    35: 9, 37: 9, 19: 9, 1: 4, 42: 2, 6: 3, 3: 2, 43: 2, 34: 99,
    2: 3, 4: 2, 5: 2, 9: 4, 10: 4, 11: 90, 12: 4, 14: 0, 18: 1, 22: 0,
    26: 0, 27: 3, 29: 6, 30: 7, 36: 9, 38: 9, 39: 3, 40: 2, 41: 2, 44: 2,
}
OCU_PADRE_POR_NIVEL = {
    35: 6, 37: 9, 19: 9, 1: 4, 42: 3, 6: 3, 3: 2, 43: 2, 34: 99,
    2: 3, 4: 2, 5: 2, 9: 7, 10: 6, 11: 0, 12: 9, 13: 90, 14: 5, 18: 1,
    20: 90, 22: 3, 25: 4, 26: 6, 27: 7, 29: 4, 30: 5, 31: 1, 33: 3,
    36: 90, 38: 9, 39: 3, 40: 2, 41: 2, 44: 2,
}

# Formación previa del estudiante (términos peruanos) -> código UCI real.
OPC_FORMACION_PREVIA = [
    ("No, solo terminé el colegio", 1),
    ("Sí, en un instituto técnico", 42),
    ("Sí, otra carrera universitaria", 2),
]

# -----------------------------------------------------------------------------
# 3. CONTEXTO ECONÓMICO PERUANO (parámetro de la APLICACIÓN, no del dataset)
#    El RMV (sueldo mínimo) peruano es S/ 1,130. El modelo se entrenó con tasas
#    macro portuguesas (desempleo, inflación, PBI); la selección del nivel de
#    ingreso familiar se traduce a un rango razonable de esas 3 variables solo
#    como parámetro de contexto de la aplicación.
# -----------------------------------------------------------------------------
NIVELES_INGRESO = [
    {
        "etiqueta": "Hasta 1 RMV (≤ S/ 1,130)",
        "unemployment": 13.0, "inflation": 3.5, "gdp": 1.5,
    },
    {
        "etiqueta": "1 a 2 RMV (S/ 1,130 – S/ 2,260)",
        "unemployment": 10.0, "inflation": 3.1, "gdp": 2.5,
    },
    {
        "etiqueta": "Más de 2 RMV (> S/ 2,260)",
        "unemployment": 7.0, "inflation": 2.7, "gdp": 3.8,
    },
]

def contexto_ingreso_a_macro(nivel):
    """
    TRANSFORMACIÓN DOCUMENTADA (requisito 13):
    Convierte el nivel de ingreso familiar seleccionado en la interfaz peruana
    a las 3 variables macro ORIGINALES del modelo (Unemployment rate, Inflation
    rate, GDP). Esta asignación es un PARÁMETRO DE CONTEXTO DE LA APLICACIÓN;
    no significa que el valor corresponda a la realidad peruana ni al dataset.
    """
    return {
        "Unemployment rate": float(nivel["unemployment"]),
        "Inflation rate": float(nivel["inflation"]),
        "GDP": float(nivel["gdp"]),
    }


def peru_nota_a_uci(nota_peruana):
    """
    TRANSFORMACIÓN DOCUMENTADA (requisito 13):
    Nota peruana 0-20 -> escala del modelo (0-200). Se usa para:
      - 'Admission grade' (originariamente 0-200 en el dataset)
      - 'Previous qualification (grade)' (originariamente 0-200)
    """
    return round(float(nota_peruana) * 10.0, 1)

# -----------------------------------------------------------------------------
# 4. TRADUCCIÓN DE VARIABLES AL ESPAÑOL (solo para etiquetas de la interfaz;
#    los nombres originales de las columnas del modelo NO se modifican)
# -----------------------------------------------------------------------------
NOMBRES_ES = {
    "Previous qualification (grade)": "Nota de formación previa",
    "Admission grade": "Nota de admisión",
    "Age at enrollment": "Edad al matricularse",
    "Curricular units 1st sem (credited)": "Cursos acreditados — Ciclo 1",
    "Curricular units 1st sem (enrolled)": "Cursos matriculados — Ciclo 1",
    "Curricular units 1st sem (evaluations)": "Cursos evaluados — Ciclo 1",
    "Curricular units 1st sem (approved)": "Cursos aprobados — Ciclo 1",
    "Curricular units 1st sem (grade)": "Nota promedio — Ciclo 1",
    "Curricular units 1st sem (without evaluations)": "Cursos sin evaluación — Ciclo 1",
    "Curricular units 2nd sem (credited)": "Cursos acreditados — Ciclo 2",
    "Curricular units 2nd sem (enrolled)": "Cursos matriculados — Ciclo 2",
    "Curricular units 2nd sem (evaluations)": "Cursos evaluados — Ciclo 2",
    "Curricular units 2nd sem (approved)": "Cursos aprobados — Ciclo 2",
    "Curricular units 2nd sem (grade)": "Nota promedio — Ciclo 2",
    "Curricular units 2nd sem (without evaluations)": "Cursos sin evaluación — Ciclo 2",
    "Unemployment rate": "Tasa de desempleo (%)",
    "Inflation rate": "Tasa de inflación (%)",
    "GDP": "PBI",
    "Marital status": "Estado civil",
    "Application mode": "Modalidad de admisión",
    "Application order": "Orden de elección de carrera",
    "Daytime/evening attendance": "Turno de estudios",
    "Previous qualification": "Formación académica previa",
    "Nacionality": "Nacionalidad",
    "Mother's qualification": "Nivel educativo de la madre",
    "Father's qualification": "Nivel educativo del padre",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Displaced": "Desplazado/a",
    "Educational special needs": "Necesidades educativas especiales",
    "Debtor": "Deudor/a de matrícula",
    "Tuition fees up to date": "Pensiones al día",
    "Gender": "Género",
    "Scholarship holder": "Becario/a",
    "International": "Estudiante internacional",
    "Course": "Carrera",
}


def traducir_feature(nombre):
    """Traduce un nombre de variable del pipeline (inglés) al español."""
    base = nombre.split("__", 1)[-1]      # quita num__ / cat__
    valor = None
    nombre_sin_valor = base
    partes = base.rsplit("_", 1)
    if len(partes) == 2 and partes[0] in NOMBRES_ES:
        nombre_sin_valor, valor = partes[0], partes[1]
    etiqueta = NOMBRES_ES.get(nombre_sin_valor,
                              nombre_sin_valor.replace("_", " "))
    if valor is not None:
        etiqueta = f"{etiqueta} = {valor}"
    return etiqueta


# -----------------------------------------------------------------------------
# 5. CARGA DE ARTEFACTOS Y DATASET
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_artefactos():
    for ruta in (RUTA_MODELO, RUTA_PREPROCESADOR):
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontró '{ruta}'. Ejecuta primero: python train_model.py"
            )
    try:
        modelo = joblib.load(RUTA_MODELO)
        preprocesador = joblib.load(RUTA_PREPROCESADOR)
    except Exception as e:
        raise RuntimeError(f"Error al cargar los archivos .pkl: {e}") from e
    return modelo, preprocesador


def cargar_metricas():
    try:
        with open(RUTA_METRICAS, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cargar_dataset():
    if not os.path.exists(RUTA_DATASET):
        return None
    try:
        df = pd.read_csv(RUTA_DATASET, sep=";")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.warning(f"No se pudo leer {RUTA_DATASET}: {e}")
        return None


# -----------------------------------------------------------------------------
# 6. EXPLICABILIDAD SHAP (opcional, con fallback)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def obtener_explicador_shap(modelo, preprocesador):
    """Devuelve un TreeExplainer de SHAP con un fondo de datos del dataset."""
    if not SHAP_DISPONIBLE:
        return None, None
    df = cargar_dataset()
    if df is None:
        return None, None
    X_back = preprocesador.transform(df[COLUMNAS_TOTALES].head(300))
    explainer = shap.TreeExplainer(modelo)
    return explainer, X_back


# -----------------------------------------------------------------------------
# 7. GRAFICO GAUGE Y RECOMENDACIONES
# -----------------------------------------------------------------------------
def crear_gauge(riesgo):
    """Gauge con zonas de color: verde <30, amarillo 30-60, rojo >60."""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=riesgo,
        number={"suffix": "%", "font": {"size": 44, "color": COLOR_TEXTO}},
        title={"text": "Probabilidad estimada de deserción",
               "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLOR_TEXTO},
            "bar": {"color": COLOR_AZUL, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": "#dadce0",
            "steps": [
                {"range": [0, 30],   "color": COLOR_VERDE},
                {"range": [30, 60],  "color": COLOR_AMBAR},
                {"range": [60, 100], "color": COLOR_ROJO},
            ],
            "threshold": {
                "line": {"color": COLOR_TEXTO, "width": 4},
                "thickness": 0.8,
                "value": riesgo,
            },
        },
    ))


def generar_recomendaciones(riesgo):
    """
    Recomendaciones de SEGUIMIENTO académico general. No son una decisión
    automática sobre el estudiante ni un diagnóstico.
    """
    if riesgo >= 60:
        nivel = "ALTO"
        color = COLOR_ROJO
        texto = ("El modelo estima un nivel de riesgo alto. Se recomienda un "
                 "seguimiento cercano por parte de tutores y orientadores:")
        puntos = [
            "Programar una entrevista de seguimiento con tutoría en los próximos 30 días.",
            "Revisar el avance de cursos aprobados y notas de los primeros ciclos.",
            "Ofrecer acompañamiento académico y consejería emocional.",
            "Evaluar la situación financiera (becas, pagos de pensiones) de forma preventiva.",
        ]
    elif riesgo >= 30:
        nivel = "MEDIO"
        color = COLOR_AMBAR
        texto = ("El modelo estima un nivel de riesgo medio. Se recomienda "
                 "mantener observación periódica:")
        puntos = [
            "Seguimiento mensual de asistencia y rendimiento académico.",
            "Reforzar tutorías en las asignaturas con menor desempeño.",
            "Ofrecer asesoría financiera preventiva si el contexto lo amerita.",
        ]
    else:
        nivel = "BAJO"
        color = COLOR_VERDE
        texto = ("El modelo estima un nivel de riesgo bajo de acuerdo con los "
                 "patrones aprendidos durante el entrenamiento. Se recomienda:")
        puntos = [
            "Mantener el seguimiento académico periódico del estudiante.",
            "Fomentar el avance regular de cursos aprobados.",
            "Reevaluar el perfil con cada nuevo ciclo académico.",
        ]
    return nivel, color, texto, puntos


# -----------------------------------------------------------------------------
# 8. PESTAÑA "PREDICTOR INDIVIDUAL" (asistente de 4 pasos)
# -----------------------------------------------------------------------------
def _g(clave, default):
    """Lee un valor persistido de un widget con key en session_state."""
    return st.session_state.get(clave, default)


def _leer(clave, default):
    """Lee un valor de usuario del asistente.

    Streamlit ELIMINA el valor de un widget de session_state en cuanto ese
    widget deja de renderizarse (p. ej. al avanzar al siguiente paso del
    asistente). Por eso cada paso guarda una copia persistente bajo "_p_<clave>"
    (ver _persistir_paso). Aquí se da prioridad al valor vivo del widget
    (mientras el paso está visible) y se recae en la copia persistente.
    """
    vivo = _g(clave, None)
    if vivo is not None:
        return vivo
    return st.session_state.get("_p_" + clave, default)


def _persistir_paso(nombre_paso):
    """Copia los valores actuales de los widgets del paso a claves persistentes
    "_p_<clave>" para que no se pierdan cuando el widget deje de renderizarse.

    Las claves "_p_*" NO son claves de widget, así que Streamlit no las borra
    nunca. construir_fila()/validar_paso()/resumen leen siempre por medio de
    _leer(), que primero consulta el widget vivo y luego la copia persistente.
    """
    claves_paso = {
        "paso1": [
            ("personal_edad", 22),
            ("personal_genero", ("Femenino", 0)),
            ("personal_carrera", ("Ingeniería de Sistemas e Informática", 9119)),
            ("personal_turno", ("Diurno", 1)),
            ("personal_formacion_previa", OPC_FORMACION_PREVIA[0]),
            ("personal_desplazado", ("No", 0)),
        ],
        "paso2": [
            ("personal_nota_admision", 13.0),
            ("acad_ciclo", 2),
            ("s1_matriculadas", 6),
            ("s1_aprobadas", 5),
            ("s1_promedio", 12.0),
            ("s2_matriculadas", 6),
            ("s2_aprobadas", 5),
            ("s2_promedio", 12.0),
        ],
        "paso3": [
            ("socio_nivel_madre", ("No estudió", 35)),
            ("socio_nivel_padre", ("No estudió", 35)),
            ("socio_ingreso", NIVELES_INGRESO[1]),
            ("socio_beca", ("No", 0)),
            ("socio_pagos", ("Sí", 1)),
        ],
    }
    for clave, default in claves_paso[nombre_paso]:
        st.session_state["_p_" + clave] = _g(clave, default)


def _indice_opcion(opciones, valor):
    """Índice de la opción seleccionada dentro de `opciones` (para reinstanciar
    un selectbox/radio con el valor que el usuario ya había elegido)."""
    try:
        return opciones.index(valor) if valor in opciones else 0
    except (TypeError, ValueError):
        return 0


def validar_paso(paso):
    """Validaciones por paso (datos imposibles). Devuelve lista de errores."""
    errores = []
    if paso == 1:
        edad = _leer("personal_edad", 22)
        if not (17 <= edad <= 80):
            errores.append("La edad debe encontrarse entre 17 y 80 años.")
    elif paso == 2:
        nota = _leer("personal_nota_admision", 13.0)
        if nota < 0 or nota > 20:
            errores.append("La nota de admisión debe encontrarse entre 0 y 20.")
        for mat_key, aprob_key, prof in [("s1_matriculadas", "s1_aprobadas", "primer"),
                                         ("s2_matriculadas", "s2_aprobadas", "segundo")]:
            mat = _leer(mat_key, 6)
            apr = _leer(aprob_key, 5)
            if mat < 0 or apr < 0:
                errores.append(f"Los cursos del {prof} ciclo no pueden ser negativos.")
            if apr > mat:
                errores.append(
                    f"En el {prof} ciclo, los cursos aprobados no pueden superar "
                    "a los cursos que llevó."
                )
            nota_s = _leer(mat_key.replace("matriculadas", "promedio"), 12.0)
            if nota_s < 0 or nota_s > 20:
                errores.append(f"La nota promedio del {prof} ciclo debe "
                               "encontrarse entre 0 y 20.")
    return errores


def construir_fila():
    """Construye las 36 columnas del modelo.

    Las variables que el usuario entrega se toman de los widgets. Las que NO se
    preguntan se completan con valores documentados (modas del dataset) y se
    muestran al usuario en el detalle 'Cómo se completó el formulario'.
    """
    edad = int(_leer("personal_edad", 22))
    nivel = _leer("socio_ingreso", NIVELES_INGRESO[1])
    macro = contexto_ingreso_a_macro(nivel)

    # Turno y carrera (con variante vespertina cuando existe).
    turno = int(_codigo(_leer("personal_turno", ("Diurno", 1)), 1))
    carrera = int(_codigo(_leer("personal_carrera", 9119), 9119))
    if turno == 0 and carrera in CARRERA_VESPERTINO:
        carrera = CARRERA_VESPERTINO[carrera]

    nota_admision = peru_nota_a_uci(_leer("personal_nota_admision", 13.0))

    sems = {}
    for pref, num in (("s1", 1), ("s2", 2)):
        mat = int(_leer(f"{pref}_matriculadas", 6))
        sems[num] = {
            "mat": mat,
            "apr": int(_leer(f"{pref}_aprobadas", 5)),
            "prom": float(_leer(f"{pref}_promedio", 12.0)),
        }

    nivel_madre = int(_codigo(_leer("socio_nivel_madre", ("Secundaria completa", 1)), 1))
    nivel_padre = int(_codigo(_leer("socio_nivel_padre", ("Secundaria completa", 1)), 1))
    pagos = int(_codigo(_leer("socio_pagos", ("Sí", 1)), 1))

    fila = {
        # --- Numéricas ---
        "Previous qualification (grade)": nota_admision,   # aprox. documentada
        "Admission grade": nota_admision,
        "Age at enrollment": edad,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": sems[1]["mat"],
        "Curricular units 1st sem (evaluations)": sems[1]["mat"],
        "Curricular units 1st sem (approved)": sems[1]["apr"],
        "Curricular units 1st sem (grade)": sems[1]["prom"],
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": sems[2]["mat"],
        "Curricular units 2nd sem (evaluations)": sems[2]["mat"],
        "Curricular units 2nd sem (approved)": sems[2]["apr"],
        "Curricular units 2nd sem (grade)": sems[2]["prom"],
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": macro["Unemployment rate"],
        "Inflation rate": macro["Inflation rate"],
        "GDP": macro["GDP"],
        # --- Categóricas (códigos originales del dataset) ---
        "Marital status": 1,                                # Soltero/a (moda)
        "Application mode": 39 if edad >= 23 else 1,        # Mayores 23 / 1.ª fase
        "Application order": 0,                             # 1.ª opción
        "Daytime/evening attendance": turno,
        "Previous qualification": int(_codigo(
            _leer("personal_formacion_previa", OPC_FORMACION_PREVIA[0]), 1)),
        "Nacionality": 1,                                   # código más frecuente
        "Mother's qualification": nivel_madre,
        "Father's qualification": nivel_padre,
        "Mother's occupation": OCU_MADRE_POR_NIVEL.get(nivel_madre, 99),
        "Father's occupation": OCU_PADRE_POR_NIVEL.get(nivel_padre, 99),
        "Displaced": int(_codigo(_leer("personal_desplazado", ("No", 0)), 0)),
        "Educational special needs": 0,                     # valor típico
        "Debtor": 0 if pagos == 1 else 1,
        "Tuition fees up to date": pagos,
        "Gender": int(_codigo(_leer("personal_genero", ("Femenino", 0)), 0)),
        "Scholarship holder": int(_codigo(_leer("socio_beca", ("No", 0)), 0)),
        "International": 0,
        "Course": carrera,
    }
    return fila


def _render_paso1():
    st.subheader("Paso 1 de 3 · Sobre el estudiante")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.slider("Edad al ingresar (años)", 17, 80,
                  int(_leer("personal_edad", 22)), 1, key="personal_edad",
                  help="Edad del estudiante al ingresar a la universidad.")
        st.selectbox("Sexo", options=[("Femenino", 0), ("Masculino", 1)],
                     index=_indice_opcion([("Femenino", 0), ("Masculino", 1)],
                                          _leer("personal_genero", ("Femenino", 0))),
                     format_func=lambda x: x[0], key="personal_genero")
    with c2:
        st.selectbox(
            "Carrera a seguir",
            options=list(NOMBRES_CARRERAS_PERU.items()),
            index=_indice_opcion(
                list(NOMBRES_CARRERAS_PERU.items()),
                _leer("personal_carrera", ("Ingeniería de Sistemas e Informática", 9119))),
            format_func=lambda x: x[0], key="personal_carrera",
            help="Carrera que cursa el estudiante. Solo se ofrecen las carreras "
                 "que el modelo puede analizar (equivalentes reales del dataset).")
        st.selectbox("Turno de estudios",
                     options=[("Diurno", 1), ("Nocturno", 0)],
                     index=_indice_opcion([("Diurno", 1), ("Nocturno", 0)],
                                          _leer("personal_turno", ("Diurno", 1))),
                     format_func=lambda x: x[0], key="personal_turno")
    with c3:
        st.selectbox("¿Estudió algo antes de la universidad?",
                     options=OPC_FORMACION_PREVIA,
                     index=_indice_opcion(
                         OPC_FORMACION_PREVIA,
                         _leer("personal_formacion_previa", OPC_FORMACION_PREVIA[0])),
                     format_func=lambda x: x[0], key="personal_formacion_previa")
        st.radio("¿Se mudó de otra ciudad para estudiar?",
                 options=[("No", 0), ("Sí", 1)], horizontal=True,
                 index=_indice_opcion([("No", 0), ("Sí", 1)],
                                      _leer("personal_desplazado", ("No", 0))),
                 format_func=lambda x: x[0], key="personal_desplazado")
    st.caption("Si no está seguro de algún dato, deje el valor que aparece por "
               "defecto; el sistema lo usará y se lo mostrará en el resultado "
               "(ver detalle técnico).")
    _persistir_paso("paso1")


def _render_paso2():
    st.subheader("Paso 2 de 3 · Notas y cursos")
    ciclo = int(_leer("acad_ciclo", 2))
    prev = st.session_state.get("_ciclo_anterior_antes")
    if prev is not None and prev != ciclo:
        for k, v in (("s1_matriculadas", 6), ("s1_aprobadas", 5),
                     ("s1_promedio", 12.0), ("s2_matriculadas", 6),
                     ("s2_aprobadas", 5), ("s2_promedio", 12.0)):
            st.session_state[k] = v
            st.session_state["_p_" + k] = v
        st.info(f"Ciclo cambiado de {prev} a {ciclo}: el registro de cursos se "
                f"reinició y ahora corresponde a los ciclos {ciclo - 1} y {ciclo}.")
    st.session_state["_ciclo_anterior_antes"] = ciclo
    st.markdown(
        f'<div class="explicacion"><b>Ventana deslizante de 2 ciclos.</b> Las '
        f"carreras peruanas duran 10 ciclos y las notas van del 0 al 20. El "
        f"modelo solo usa 2 ciclos de rendimiento; para que TODOS los ciclos "
        f"cuenten, con el <b>ciclo actual (N)</b> se evalúan los "
        f"<b>ciclos {ciclo - 1} y {ciclo}</b>: los dos últimos completados.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Nota de admisión (0 – 20)", 0.0, 20.0,
                  float(_leer("personal_nota_admision", 13.0)), 0.5,
                  key="personal_nota_admision",
                  help="Nota con la que ingresó a la carrera (examen de "
                       "admisión o promedio de colegio).")
    with c2:
        st.slider("Ciclo actual (último cursado)", 2, 10,
                  int(_leer("acad_ciclo", 2)), 1, key="acad_ciclo",
                  help="Indica en qué ciclo está el estudiante. Se evalúan los "
                       f"dos ciclos más recientes (ciclos {ciclo - 1} y "
                       f"{ciclo}). Se necesita al menos ciclo 2 por el modelo.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"**Ciclo {ciclo - 1}**")
        st.number_input(f"Cursos que llevó (ciclo {ciclo - 1})", 0, 20,
                        int(_leer("s1_matriculadas", 6)), 1, key="s1_matriculadas")
        st.number_input(f"Cursos que aprobó (ciclo {ciclo - 1})", 0, 20,
                        int(_leer("s1_aprobadas", 5)), 1, key="s1_aprobadas",
                        help="Debe ser menor o igual a los cursos que llevó.")
        st.number_input(f"Promedio de notas 0-20 (ciclo {ciclo - 1})", 0.0, 20.0,
                        float(_leer("s1_promedio", 12.0)), 0.1, key="s1_promedio")
    with c4:
        st.markdown(f"**Ciclo {ciclo}**")
        st.number_input(f"Cursos que llevó (ciclo {ciclo})", 0, 20,
                        int(_leer("s2_matriculadas", 6)), 1, key="s2_matriculadas")
        st.number_input(f"Cursos que aprobó (ciclo {ciclo})", 0, 20,
                        int(_leer("s2_aprobadas", 5)), 1, key="s2_aprobadas")
        st.number_input(f"Promedio de notas 0-20 (ciclo {ciclo})", 0.0, 20.0,
                        float(_leer("s2_promedio", 12.0)), 0.1, key="s2_promedio")
    _persistir_paso("paso2")


def _render_paso3():
    st.subheader("Paso 3 de 3 · Familia y economía")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Nivel de estudios de la madre",
                     options=list(NIVELES_EDU_PADRE_PERU.items()),
                     index=_indice_opcion(
                         list(NIVELES_EDU_PADRE_PERU.items()),
                         _leer("socio_nivel_madre", ("No estudió", 35))),
                     format_func=lambda x: x[0], key="socio_nivel_madre")
        st.selectbox("Nivel de estudios del padre",
                     options=list(NIVELES_EDU_PADRE_PERU.items()),
                     index=_indice_opcion(
                         list(NIVELES_EDU_PADRE_PERU.items()),
                         _leer("socio_nivel_padre", ("No estudió", 35))),
                     format_func=lambda x: x[0], key="socio_nivel_padre")
        st.caption("La ocupación de los padres se estima según su nivel de "
                   "estudios (el más frecuente en los datos del modelo).")
    with c2:
        st.selectbox("Situación económica familiar (RMV Perú: S/ 1,130)",
                     options=NIVELES_INGRESO,
                     index=_indice_opcion(
                         NIVELES_INGRESO,
                         _leer("socio_ingreso", NIVELES_INGRESO[1])),
                     format_func=lambda n: n["etiqueta"], key="socio_ingreso",
                     help="Ingresos del hogar comparados con el sueldo mínimo "
                          "peruano vigente (RMV = S/ 1,130).")
        st.radio("¿Tiene beca o media beca?", options=[("No", 0), ("Sí", 1)],
                 horizontal=True,
                 index=_indice_opcion([("No", 0), ("Sí", 1)],
                                      _leer("socio_beca", ("No", 0))),
                 format_func=lambda x: x[0], key="socio_beca",
                 help="Beca 18, PRONABEC, beca interna de la universidad u otra.")
        st.radio("¿Está al día con las pensiones?",
                 options=[("Sí", 1), ("No", 0)], horizontal=True,
                 index=_indice_opcion([("Sí", 1), ("No", 0)],
                                      _leer("socio_pagos", ("Sí", 1))),
                 format_func=lambda x: x[0], key="socio_pagos")
    _persistir_paso("paso3")


def _mostrar_shap(modelo, preprocesador, fila):
    """Contribuciones SHAP de la predicción (con fallback a importancia global)."""
    st.markdown(
        '<div class="explicacion"><b>Factores asociados:</b> la lista siguiente '
        "muestra las variables con mayor contribución al resultado del modelo. "
        "Estas NO causan la deserción; son variables estadísticamente asociadas "
        "a la predicción según los patrones aprendidos.</div>",
        unsafe_allow_html=True,
    )
    try:
        explainer, X_back = obtener_explicador_shap(modelo, preprocesador)
        if explainer is None:
            raise ValueError("SHAP no disponible")
        X_row = preprocesador.transform(pd.DataFrame([fila]))
        valores = explainer.shap_values(X_row)
        if isinstance(valores, list):
            valores = valores[1]
        contrib = np.asarray(valores)[0]
        nombres = list(preprocesador.get_feature_names_out())
        df_contrib = pd.DataFrame({
            "variable": [traducir_feature(n) for n in nombres],
            "cambio_log_odds": contrib,
            "magnitud": np.abs(contrib),
        }).sort_values("magnitud", ascending=False).head(8)
        fig = px.bar(
            df_contrib, x="cambio_log_odds", y="variable", orientation="h",
            color=np.where(df_contrib["cambio_log_odds"] >= 0, "sube riesgo",
                           "baja riesgo"),
            color_discrete_map={"sube riesgo": COLOR_ROJO,
                                "baja riesgo": COLOR_VERDE},
            title="Contribución de cada variable a la predicción (SHAP)",
            labels={"cambio_log_odds": "Cambio en log-odds (hacia Dropout)",
                    "variable": "Variable"},
        )
        fig.update_layout(height=360, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Rojo = empuja la predicción hacia deserción. Verde = la "
                   "reduce. Es una explicación local de esta predicción.")
    except Exception:
        imp = pd.Series(modelo.feature_importances_,
                        index=[traducir_feature(n)
                               for n in preprocesador.get_feature_names_out()])
        imp = imp.sort_values().tail(8)
        fig = px.bar(
            imp, x=imp.values, y=imp.index, orientation="h",
            title="Variables más influyentes (importancia global del modelo)",
            labels={"x": "Importancia relativa", "y": "Variable"},
        )
        fig.update_layout(height=360, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("SHAP no disponible; se muestra la importancia global "
                   "promedio del modelo XGBoost.")


def _mostrar_como_se_completo():
    """Explica de forma VISIBLE qué variables se completaron y con qué valor."""
    st.markdown("**Cómo se completó el formulario**")
    edad = int(_leer("personal_edad", 22))
    nivel_madre = int(_codigo(_leer("socio_nivel_madre", ("Secundaria completa", 1)), 1))
    nivel_padre = int(_codigo(_leer("socio_nivel_padre", ("Secundaria completa", 1)), 1))
    modo = "39 (Mayores de 23 años)" if edad >= 23 else "1 (1.ª fase general)"
    ocup_madre = OCU_MADRE_POR_NIVEL.get(nivel_madre, 99)
    ocup_padre = OCU_PADRE_POR_NIVEL.get(nivel_padre, 99)
    filas = [
        ("Ventana deslizante de ciclos",
         "El modelo solo usa 2 ciclos de rendimiento; con el ciclo actual N "
         "se evalúan los ciclos N-1 y N (los dos últimos completados), de "
         "modo que cualquier momento de la carrera de 10 ciclos genera "
         "riesgo."),
        ("Estado civil",
         "Soltero/a (código 1) — el más frecuente del dataset (88.6 %)."),
        ("Nacionalidad",
         "Código 1 del dataset (portugués original) — el más frecuente "
         "(97.5 %). No sustituye la nacionalidad real del estudiante."),
        ("Modalidad de ingreso",
         f"{modo} — seleccionada según la edad (en el dataset, los mayores "
         "de 23 años ingresan por la modalidad 39)."),
        ("Orden de elección de la carrera", "0 (primera opción) — valor típico."),
        ("Nota de formación previa",
         "Se usa la nota de admisión como aproximación (en el dataset, "
         "Previous qualification y Admission grade correlacionan 0.58)."),
        ("Ocupación de la madre",
         f"Estimada según su nivel de estudios (la más frecuente del "
         f"dataset): {_nombre(NOMBRES_OCU_MADRE, ocup_madre)}."),
        ("Ocupación del padre",
         f"Estimada según su nivel de estudios (la más frecuente del "
         f"dataset): {_nombre(NOMBRES_OCU_PADRE, ocup_padre)}."),
        ("Cursos evaluados (de los 2 ciclos de la ventana)",
         "Igual a los cursos que llevó (supuesto: todas las asignaturas "
         "matriculadas se evalúan)."),
        ("Cursos acreditados y sin evaluación",
         "0 en los dos primeros ciclos — lo más frecuente al ingresar "
         "(87-93 % del dataset)."),
        ("Estudiante internacional / con necesidades especiales",
         "No — valor típico del dataset."),
    ]
    for titulo, detalle in filas:
        st.markdown(f"- **{titulo}:** {detalle}")


def _render_resultado(modelo, preprocesador):
    res = st.session_state.get("_resultado")
    if res is None:
        st.info("Complete el formulario y pulse 'Predecir riesgo'.")
        return

    riesgo = res["riesgo"]
    nivel, color, texto, puntos = res["recomendacion"]
    fila = res["fila"]
    resumen = res["resumen"]

    st.markdown("## Resultado de la predicción")
    c_gauge, c_info = st.columns([1, 1.4])

    with c_gauge:
        st.plotly_chart(crear_gauge(riesgo), width="stretch")
        st.caption(
            f"Estimación al ciclo {resumen['ciclo']} de 10 (ventana: ciclos "
            f"{resumen['c_prev']} y {resumen['c_cur']}). El valor es una "
            "ESTIMACIÓN generada por el modelo (predict_proba sobre la clase "
            "Dropout=1). No es una certeza ni una probabilidad real individual."
        )
    with c_info:
        clase_tarjeta = ("tarjeta-roja" if nivel == "ALTO"
                         else "tarjeta-ambar" if nivel == "MEDIO"
                         else "tarjeta-verde")
        st.markdown(
            f'<div class="tarjeta {clase_tarjeta}"><h3 style="margin:0;'
            f'color:{color};">Nivel de riesgo: {nivel}</h3></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Probabilidad estimada de deserción:** {riesgo:.1f} %"
        )
        st.markdown(f"**{texto}**")
        st.caption(
            "Umbrales definidos por la APLICACIÓN (0-30 % Bajo, 30-60 % Medio, "
            ">60 % Alto) solo para visualización; no son categorías aprendidas "
            "por el modelo. Clase de interés: Dropout=1 frente a "
            "Graduado/Enrolled=0."
        )

    st.markdown("---")

    # Resumen del estudiante --------------------------------------------- #
    st.subheader("Resumen del estudiante")
    resumen = res["resumen"]
    cols = st.columns(5)
    celdas = [
        ("Edad", f"{resumen['edad']} años"),
        ("Sexo", resumen["genero"]),
        ("Beca", resumen["beca"]),
        ("Ciclo actual", f"ciclo {resumen['ciclo']} de 10"),
        ("Carrera", resumen["carrera"]),
    ]
    for i, (titulo, valor) in enumerate(celdas):
        with cols[i % 5]:
            st.markdown(
                f'<div class="tarjeta tarjeta-azul"><b>{titulo}</b><br>'
                f'<span style="font-size:1.1rem;">{valor}</span></div>',
                unsafe_allow_html=True,
            )
    cols2 = st.columns(5)
    celdas2 = [
        ("Nota de admisión", f"{resumen['nota_admision']}/20"),
        (f"Promedio ciclo {resumen['c_prev']}", f"{resumen['prom_s1']}/20"),
        (f"Promedio ciclo {resumen['c_cur']}", f"{resumen['prom_s2']}/20"),
        ("Cursos aprobados", resumen["aprobados"]),
        ("Situación económica", resumen["economia"]),
    ]
    for i, (titulo, valor) in enumerate(celdas2):
        with cols2[i % 5]:
            st.markdown(
                f'<div class="tarjeta tarjeta-azul"><b>{titulo}</b><br>'
                f'<span style="font-size:1.1rem;">{valor}</span></div>',
                unsafe_allow_html=True,
            )
    st.caption(f"La suma de cursos aprobados (ciclo {resumen['c_prev']} + "
               f"ciclo {resumen['c_cur']}) es un dato RESUMEN de la "
               "aplicación; el dataset no contiene una variable de créditos "
               "acumulados por ciclo.")

    # Evolución académica ------------------------------------------------- #
    st.subheader("Evolución de notas (ciclos "
                 f"{resumen['c_prev']} y {resumen['c_cur']})")
    ev = pd.DataFrame([
        {"Ciclo": f"Ciclo {resumen['c_prev']}", "Promedio": resumen["prom_s1"],
         "Cursos aprobados": resumen["aprob_s1"]},
        {"Ciclo": f"Ciclo {resumen['c_cur']}", "Promedio": resumen["prom_s2"],
         "Cursos aprobados": resumen["aprob_s2"]},
    ])
    fig_ev = px.line(ev, x="Ciclo", y="Promedio", markers=True,
                     title="Promedio de notas por ciclo",
                     color_discrete_sequence=[COLOR_AZUL], labels={"value": "Nota 0-20"})
    fig_ev.update_traces(line_width=3)
    fig_ev.update_layout(height=300)
    st.plotly_chart(fig_ev, width="stretch")
    st.caption("Ventana deslizante: el modelo usa 2 ciclos de rendimiento; "
               f"se evalúan los ciclos {resumen['c_prev']} y "
               f"{resumen['c_cur']} (los dos más recientes del estudiante).")

    # Factores considerados ---------------------------------------------- #
    st.subheader("Factores considerados en la predicción")
    factores_clave = [
        (f"Cursos aprobados (ciclo {resumen['c_prev']})", resumen["aprob_s1"]),
        (f"Cursos aprobados (ciclo {resumen['c_cur']})", resumen["aprob_s2"]),
        (f"Nota promedio (ciclo {resumen['c_prev']})", f"{resumen['prom_s1']}/20"),
        (f"Nota promedio (ciclo {resumen['c_cur']})", f"{resumen['prom_s2']}/20"),
        (f"Cursos que llevó (ciclo {resumen['c_prev']})", fila["Curricular units 1st sem (enrolled)"]),
        (f"Cursos que llevó (ciclo {resumen['c_cur']})", fila["Curricular units 2nd sem (enrolled)"]),
        ("Nota de admisión", f"{resumen['nota_admision']}/20"),
        ("Beca", resumen["beca"]),
        ("Situación económica", resumen["economia"]),
        ("Género", resumen["genero"]),
        ("Edad", resumen["edad"]),
        ("Carrera", resumen["carrera"]),
    ]
    df_fact = pd.DataFrame(factores_clave, columns=["Variable", "Valor usado por el modelo"])
    df_fact["Valor usado por el modelo"] = df_fact["Valor usado por el modelo"].astype(str)
    st.dataframe(df_fact, width="stretch", hide_index=True)
    st.markdown(
        '<div class="recomendacion"><b>Nota:</b> el formulario pide solo la '
        "información que un tutor puede obtener con facilidad. Las demás "
        "variables que el modelo necesita se completan con valores "
        "documentados (los más frecuentes en el dataset) y se muestran en "
        "«Ver detalle técnico». Se presume que todos los cursos matriculados "
        "fueron evaluados (evaluaciones = matriculados). Las únicas variables "
        "que llegan por transformación de contexto son Unemployment rate, "
        "Inflation rate y GDP (asignadas según la situación económica "
        "familiar seleccionada).</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Ver detalle técnico (36 variables del modelo)"):
        _mostrar_como_se_completo()
        st.markdown("---")
        df_full = pd.DataFrame([_nombre_categorias(fila)]).T.rename(
            columns={0: "Valor"})
        df_full["Valor"] = df_full["Valor"].astype(str)
        st.dataframe(df_full, width="stretch")

    _mostrar_shap(modelo, preprocesador, fila)

    # Recomendaciones ---------------------------------------------------- #
    st.subheader("Recomendación de seguimiento")
    for p in puntos:
        st.markdown(f"- ✅ {p}")
    st.markdown(
        '<div class="recomendacion"><b>Aviso:</b> las recomendaciones son de '
        "seguimiento académico general y NO constituyen una decisión automática "
        "sobre el estudiante. El resultado es una ESTIMACIÓN; no reemplaza la "
        "evaluación de docentes, tutores ni orientadores.</div>",
        unsafe_allow_html=True,
    )


def render_predictor(modelo, preprocesador):
    st.title("📝 Predictor Individual")
    st.markdown(
        "<h4>Adaptación / propuesta de aplicación de un modelo de Machine "
        "Learning al contexto universitario de Trujillo</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="recomendacion"><b>Importante (adaptación peruana):</b> '
        "el modelo fue entrenado con el dataset "
        "'Predict Students' Dropout and Academic Success' del UCI ML "
        "Repository. No fue recolectado en Perú, por lo que esta herramienta "
        "NO afirma representar estadísticamente a todos los estudiantes "
        "peruanos: es una propuesta de aplicación adaptada a notas 0-20, "
        "carreras de 10 ciclos y contexto económico local.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Sobre el modelo"):
        st.markdown(
            "El sistema utiliza un modelo de Machine Learning entrenado con el "
            "dataset *Predict Students' Dropout and Academic Success* del "
            "repositorio UCI. El modelo identifica patrones asociados con la "
            "deserción y el éxito académico a partir de variables académicas, "
            "demográficas y socioeconómicas.\n\n"
            "**El resultado es una estimación y no constituye un diagnóstico "
            "definitivo ni reemplaza la evaluación de docentes, tutores u "
            "orientadores.**"
        )

    paso = int(_g("_paso", 1))
    st.markdown(
        f'<div class="paso-indicador">Paso: '
        f"{'1/3 Sobre el estudiante' if paso <= 1 else '2/3 Notas y cursos' if paso == 2 else '3/3 Familia y economía' if paso == 3 else 'Resultado'}"
        "</div>",
        unsafe_allow_html=True,
    )

    if paso == 1:
        _render_paso1()
    elif paso == 2:
        _render_paso2()
    elif paso == 3:
        _render_paso3()
    else:
        _render_resultado(modelo, preprocesador)

    # Botones de navegación del asistente -------------------------------- #
    st.markdown("---")
    b1, b2, b3 = st.columns([1, 1, 1])

    if paso > 1:
        with b1:
            if st.button("⬅ Anterior", width="stretch"):
                st.session_state["_paso"] = max(paso - 1, 1)
                st.rerun()

    if paso < 3:
        with b2:
            if st.button("Siguiente ➡", width="stretch", type="primary"):
                errores = validar_paso(paso)
                if errores:
                    for e in errores:
                        st.error(e)
                else:
                    st.session_state["_paso"] = paso + 1
                    st.rerun()
    elif paso == 3:
        with b2:
            if st.button("🔮 Predecir riesgo", width="stretch",
                         type="primary"):
                errores = validar_paso(3)
                if errores:
                    for e in errores:
                        st.error(e)
                else:
                    fila = construir_fila()
                    try:
                        X_t = preprocesador.transform(pd.DataFrame([fila]))
                        idx_drop = int(np.where(modelo.classes_ == 1)[0][0])
                        proba = float(modelo.predict_proba(X_t)[0][idx_drop])
                        riesgo = min(proba * 100.0, 100.0)
                        nivel, color, texto, puntos = generar_recomendaciones(riesgo)
                        resumen = _construir_resumen(fila)
                        st.session_state["_resultado"] = {
                            "riesgo": riesgo,
                            "fila": fila,
                            "resumen": resumen,
                            "recomendacion": (nivel, color, texto, puntos),
                        }
                        st.session_state["_paso"] = 4
                        st.rerun()
                    except Exception as e:
                        st.error(f"No se pudo realizar la predicción: {e}")
                        st.code(traceback.format_exc())

    if paso == 4:
        with b3:
            if st.button("🔄 Nueva consulta", width="stretch"):
                st.session_state["_paso"] = 1
                for k in list(st.session_state.keys()):
                    if k.startswith(("personal_", "socio_", "s1_", "s2_",
                                     "acad_", "_resultado", "_p_")):
                        st.session_state.pop(k, None)
                st.session_state.pop("_ciclo_anterior_antes", None)
                st.rerun()


def _construir_resumen(fila):
    """Construye el resumen amigable a partir de la fila ya transformada."""
    beca = "Sí" if fila["Scholarship holder"] == 1 else "No"
    gen = "Masculino" if fila["Gender"] == 1 else "Femenino"
    nivel = _leer("socio_ingreso", NIVELES_INGRESO[1])
    carrera = NOMBRE_CARRERA_POR_CODIGO.get(
        int(fila["Course"]), f"Carrera {int(fila['Course'])}")
    ciclo = int(_leer("acad_ciclo", 2))
    return {
        "edad": int(fila["Age at enrollment"]),
        "genero": gen,
        "beca": beca,
        "ciclo": ciclo,
        "c_prev": ciclo - 1,
        "c_cur": ciclo,
        "carrera": carrera,
        "nota_admision": round(fila["Admission grade"] / 10.0, 1),
        "prom_s1": round(float(fila["Curricular units 1st sem (grade)"]), 1),
        "prom_s2": round(float(fila["Curricular units 2nd sem (grade)"]), 1),
        "aprob_s1": int(fila["Curricular units 1st sem (approved)"]),
        "aprob_s2": int(fila["Curricular units 2nd sem (approved)"]),
        "aprobados": f"{int(fila['Curricular units 1st sem (approved)']) + int(fila['Curricular units 2nd sem (approved)'])} (ciclo {ciclo - 1} + ciclo {ciclo})",
        "economia": nivel["etiqueta"],
    }


# -----------------------------------------------------------------------------
# 9. PESTAÑA "INICIO"
# -----------------------------------------------------------------------------
def render_inicio(metricas):
    st.title("🎓 EduPredict Trujillo")
    st.markdown(
        "<h4>Sistema predictivo de deserción universitaria basado en Machine "
        "Learning · Propuesta de aplicación adaptada al contexto peruano</h4>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="tarjeta tarjeta-azul"><b>ODS 4</b><br>'
            "Educación de calidad · Meta 4.3, acceso igualitario a la "
            "educación superior.</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="tarjeta tarjeta-azul"><b>Algoritmo</b><br>'
            "XGBoost + SMOTE (balanceo SOLO en entrenamiento). "
            "36 variables reales del dataset UCI.</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="tarjeta tarjeta-azul"><b>Adaptado a Perú</b><br>'
            "Notas 0-20, carreras de 10 ciclos y RMV S/ 1,130 como parámetro "
            "de contexto de la aplicación.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Metodología del sistema")

    with st.expander("ℹ️ Sobre el modelo"):
        st.markdown(
            "El sistema utiliza un modelo de Machine Learning entrenado con el "
            "dataset *Predict Students' Dropout and Academic Success* del "
            "repositorio UCI. El modelo identifica patrones asociados con la "
            "deserción y el éxito académico a partir de variables académicas, "
            "demográficas y socioeconómicas.\n\n"
            "**El resultado es una estimación y no constituye un diagnóstico "
            "definitivo ni reemplaza la evaluación de docentes, tutores u "
            "orientadores.**"
        )

    st.markdown(
        f"""
        **Interpretación de las clases del modelo:**

        | Clase original | Valor binarizado | Significado en el sistema |
        |----------------|------------------|---------------------------|
        | Dropout | `1` | Desertor (clase de interés; 'riesgo de deserción') |
        | Graduate / Enrolled | `0` | Se gradúa o continúa matriculado |

        El **% de riesgo** mostrado es `predict_proba()` sobre la clase `1`
        (Dropout). Es una estimación, no una probabilidad real individual.

        **Pipeline de entrenamiento (libre de data leakage):**

        1. `train_test_split` estratificado 80/20 (`random_state=42`).
        2. `ColumnTransformer`: `StandardScaler` (numéricas) + `OneHotEncoder`
           (`handle_unknown="ignore"`) ajustado **solo en entrenamiento**.
        3. **SMOTE dentro del pipeline** (`imblearn`): actúa únicamente sobre
           el entrenamiento; el test jamás se balancea ni es visto por SMOTE.
        4. `XGBClassifier` (n_estimators=200, learning_rate=0.1, max_depth=6).
        5. Evaluación sobre el test original (sin SMOTE).

        **Umbrales de riesgo:** BAJO 0-30 %, MEDIO 30-60 %, ALTO >60 %. Son
        definidos por la APLICACIÓN para la visualización; no son categorías
        aprendidas por el modelo.
        """
    )

    if metricas:
        m = metricas
        st.subheader("Rendimiento del modelo entrenado (sobre el test, sin SMOTE)")
        col = st.columns(5)
        col[0].metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
        col[1].metric("Precision (Dropout)", f"{m.get('precision_dropout', 0):.2%}")
        col[2].metric("Recall (Dropout)", f"{m.get('recall_dropout', 0):.2%}")
        col[3].metric("F1-Score (Dropout)", f"{m.get('f1_dropout', 0):.2%}")
        col[4].metric("AUC-ROC", f"{m.get('auc_roc', 0):.2%}")
        st.caption(m.get("interpretacion_clases", ""))

    st.markdown("---")
    st.subheader("Créditos y contexto")
    st.markdown(
        f"""
        - **Proyecto TIF Nivel III** — Universidad César Vallejo, Trujillo (2026).
        - **Dataset:** [Predict students' dropout and academic success]({URL_UCI}),
          UCI ML Repository (licencia CC BY 4.0).
        - **Adaptación:** las notas 0-20 y el RMV S/ 1,130 son parámetros de la
          aplicación; el modelo conserva las variables y escalas originales.
        """
    )


# -----------------------------------------------------------------------------
# 10. PESTAÑA "ANÁLISIS DE DATOS"
# -----------------------------------------------------------------------------
def render_analisis(modelo, preprocesador):
    st.title("📊 Análisis de Datos")
    st.markdown(
        "Análisis exploratorio del dataset y de la importancia de las variables "
        "del modelo, **interpretado para entender por qué un estudiante abandona "
        "la universidad**. Coloca `data.csv` en la carpeta del proyecto para "
        "activar todos los gráficos."
    )
    st.markdown("---")

    # 10.1 Métricas y matriz de confusión (desde metricas.json)
    metricas = cargar_metricas()
    if metricas:
        st.subheader("Métricas del modelo y matriz de confusión")
        m = metricas
        col = st.columns(5)
        col[0].metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
        col[1].metric("Precision (Dropout)", f"{m.get('precision_dropout', 0):.2%}")
        col[2].metric("Recall (Dropout)", f"{m.get('recall_dropout', 0):.2%}")
        col[3].metric("F1-Score (Dropout)", f"{m.get('f1_dropout', 0):.2%}")
        col[4].metric("AUC-ROC", f"{m.get('auc_roc', 0):.2%}")
        if "confusion_matrix" in m:
            cm = np.array(m["confusion_matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True, aspect="auto",
                color_continuous_scale="Blues",
                title="Matriz de confusión (test, sin SMOTE)",
                labels={"x": "Predicción", "y": "Real"},
                x=["Graduado/Enrolled (0)", "Dropout (1)"],
                y=["Graduado/Enrolled (0)", "Dropout (1)"],
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, width="stretch")
            st.caption("Fila = valor real, columna = predicción del modelo.")
        if "distribucion_binarizada" in m:
            st.caption(
                "Distribución de clases binarizada: "
                f"Dropout(1) = {m['distribucion_binarizada'].get('Dropout(1)', '')}, "
                f"Graduado/Enrolled(0) = "
                f"{m['distribucion_binarizada'].get('Graduado/Enrolled(0)', '')}. "
                "El desequilibrio justifica el uso de SMOTE solo en el "
                "entrenamiento."
            )
        st.markdown("---")

    # 10.2 Importancia de las características
    st.subheader("Importancia de las características (modelo XGBoost)")
    st.markdown(
        '<div class="explicacion"><b>¿Qué muestra?</b> Qué tanto "decide" cada '
        "variable dentro del modelo XGBoost para separar a los desertores de los "
        "que egresan. <b>¿Por qué importa?</b> Permite saber DÓNDE enfocar la "
        "prevención: las variables de rendimiento de los primeros ciclos suelen "
        "mandar.</div>",
        unsafe_allow_html=True,
    )
    try:
        nombres = list(preprocesador.get_feature_names_out())
        imp = pd.Series(modelo.feature_importances_, index=nombres)
        imp = imp.sort_values().tail(12)
        imp.index = [traducir_feature(n) for n in imp.index]
        fig_imp = px.bar(
            imp,
            x=imp.values,
            y=imp.index,
            orientation="h",
            color=imp.values,
            color_continuous_scale=[COLOR_VERDE, COLOR_AMBAR, COLOR_ROJO],
            title="Top 12 variables según importancia del modelo",
            labels={"x": "Importancia relativa", "y": "Variable"},
        )
        fig_imp.update_layout(coloraxis_showscale=False, height=460)
        st.plotly_chart(fig_imp, width="stretch")
    except Exception:
        st.info("No se pudo calcular la importancia de características.")

    df = cargar_dataset()
    if df is None:
        st.warning(
            f"No se encontró `{RUTA_DATASET}` para el análisis exploratorio. "
            "Descárgalo del UCI ML Repository y reinicia la app."
        )
        return

    df.columns = [c.strip() for c in df.columns]
    df["Target"] = df["Target"].astype(str).str.strip().map(
        {"Dropout": "Abandonó", "Graduate": "Graduado", "Enrolled": "Matriculado"}
    )
    df["Nota de admisión (0-20)"] = df["Admission grade"] / 10.0

    # 10.3 KPIs generales
    n_dropout = int((df["Target"] == "Abandonó").sum())
    st.subheader("KPIs del dataset (UCI)")
    st.markdown(
        '<div class="explicacion"><b>¿Qué son estos KPIs?</b> Los números base '
        "del dataset sobre el que se entrenó el modelo. <b>¿Por qué importan?</b> "
        "Muestran que la deserción es un problema real (~1 de cada 3 estudiantes "
        "abandona) y que el modelo se construyó sin datos faltantes.",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total estudiantes", len(df))
    c2.metric("Variables predictoras", len(COLUMNAS_TOTALES))
    c3.metric("Desertores (Abandonó)", f"{n_dropout} ({n_dropout / len(df):.1%})")
    c4.metric("Datos faltantes", int(df.isna().sum().sum()))

    # 10.4 Distribución de la variable objetivo
    st.subheader("Distribución de la variable objetivo")
    st.markdown(
        '<div class="explicacion"><b>¿Qué muestra?</b> Cuántos estudiantes de la '
        "muestra terminaron en cada estado. <b>¿Por qué importa?</b> Confirma el "
        "desequilibrio de clases; por eso el modelo usa SMOTE (solo en "
        "entrenamiento) y se evalúa con Recall/F1.",
        unsafe_allow_html=True,
    )
    conteo = df["Target"].value_counts().reset_index()
    conteo.columns = ["Target", "Conteo"]
    colores = {
        "Graduado": COLOR_VERDE, "Matriculado": COLOR_AZUL, "Abandonó": COLOR_ROJO,
    }
    fig_target = px.bar(
        conteo, x="Target", y="Conteo", color="Target",
        color_discrete_map=colores,
        text="Conteo", title="Estudiantes por estado final",
    )
    fig_target.update_traces(textposition="outside")
    fig_target.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_target, width="stretch")

    # 10.5 Histograma de la nota de admisión por estado
    st.subheader("Nota de admisión vs estado final")
    st.markdown(
        '<div class="explicacion"><b>¿Qué muestra?</b> La distribución de la nota '
        "de ingreso (escala 0-20) de quienes abandonaron, se graduaron o siguen "
        "matriculados. <b>¿Por qué importa?</b> El grupo que abandona suele tener "
        "notas de ingreso más bajas y dispersas.",
        unsafe_allow_html=True,
    )
    fig_nota = px.histogram(
        df, x="Nota de admisión (0-20)", color="Target", nbins=40,
        color_discrete_map=colores, barmode="overlay",
        title="Nota de admisión (escala 0-20) por estado final",
        opacity=0.7,
    )
    fig_nota.update_layout(height=380)
    st.plotly_chart(fig_nota, width="stretch")

    # 10.6 Aprobados primer ciclo (boxplot)
    st.subheader("Unidades aprobadas en el 1er ciclo vs estado final")
    st.markdown(
        '<div class="explicacion"><b>Recordatorio:</b> las <b>unidades '
        "aprobadas</b> son los CURSOS aprobados (de los ~6 matriculados por "
        "ciclo). <b>¿Por qué importa?</b> Desaprobar varios cursos en el primer "
        "ciclo es un fuerte predictor de abandono.",
        unsafe_allow_html=True,
    )
    fig_aprob = px.box(
        df, x="Target",
        y="Curricular units 1st sem (approved)",
        color="Target", color_discrete_map=colores,
        title="Cursos aprobados en el 1er ciclo por estado final",
        labels={
            "Target": "Estado final",
            "Curricular units 1st sem (approved)": "Cursos aprobados (ciclo 1)",
        },
    )
    fig_aprob.update_layout(yaxis_title="Cursos aprobados (ciclo 1)", height=400,
                            showlegend=False)
    st.plotly_chart(fig_aprob, width="stretch")

    # 10.7 Matriz de correlación
    st.subheader("Matriz de correlación (variables numéricas)")
    st.markdown(
        '<div class="explicacion"><b>¿Qué muestra?</b> Coeficiente de Pearson '
        "entre variables numéricas (1 = relación directa perfecta, -1 = inversa, "
        "0 = sin relación). <b>¿Por qué importa?</b> Identifica patrones "
        "(más cursos aprobados se asocia a mejores notas) y redundancias.",
        unsafe_allow_html=True,
    )
    correlacion_cols = [
        "Admission grade", "Previous qualification (grade)",
        "Age at enrollment", "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Unemployment rate", "Inflation rate", "GDP",
    ]
    correlacion_cols = [c for c in correlacion_cols if c in df.columns]
    df_corr = df[correlacion_cols].rename(columns=NOMBRES_ES)
    corr = df_corr.corr()
    corr.index = [traducir_feature(c) for c in corr.index]
    corr.columns = [traducir_feature(c) for c in corr.columns]
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación de Pearson entre variables numéricas",
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, width="stretch")

    st.caption(
        "Estos análisis describen el dataset ORIGINAL (recolectado en Portugal). "
        "Sirven de referencia metodológica para la propuesta aplicada a Trujillo."
    )


# -----------------------------------------------------------------------------
# 11. PUNTO DE ENTRADA DE LA APP
# -----------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.markdown("<div style='font-size:3rem;'>🎓</div>", unsafe_allow_html=True)
        st.title("EduPredict Trujillo")
        st.caption("UCV Trujillo · TIF Nivel III · 2026")
        st.markdown("---")
        st.markdown("**Paleta de riesgo**")
        st.markdown(
            f'<span style="color:{COLOR_VERDE};">● Bajo</span> '
            f'<span style="color:{COLOR_AMBAR};">● Medio</span> '
            f'<span style="color:{COLOR_ROJO};">● Alto</span>',
            unsafe_allow_html=True,
        )
        st.caption("Umbrales 0-30 / 30-60 / >60 % definidos por la aplicación.")
        st.markdown(
            f"**Dataset:** UCI ML Repository (adaptado a Perú: notas 0-20, "
            f"10 ciclos, RMV S/ 1,130)",
            unsafe_allow_html=True,
        )
        st.markdown("**ODS:** Objetivo 4 · Meta 4.3")

    try:
        modelo, preprocesador = cargar_artefactos()
    except Exception as e:
        st.error("🚨 No se pudieron cargar los modelos entrenados.")
        st.code(f"{e}\n\n{traceback.format_exc()}")
        st.markdown(
            "**Solución:** descarga `data.csv` del UCI ML Repository, colócalo "
            "en esta carpeta y ejecuta `python train_model.py`."
        )
        st.stop()

    metricas = cargar_metricas()

    pestaña_inicio, pestaña_predictor, pestaña_analisis = st.tabs(
        ["🏠 Inicio", "📝 Predictor Individual", "📊 Análisis de Datos"]
    )

    with pestaña_inicio:
        render_inicio(metricas)

    with pestaña_predictor:
        st.session_state.setdefault("_paso", 1)
        render_predictor(modelo, preprocesador)

    with pestaña_analisis:
        render_analisis(modelo, preprocesador)


if __name__ == "__main__":
    main()