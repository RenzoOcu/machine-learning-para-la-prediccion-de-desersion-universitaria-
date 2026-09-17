# -*- coding: utf-8 -*-
"""
===============================================================================
 EduPredict Trujillo - API Serverless FastAPI + Dashboard Web para Vercel
===============================================================================
  Adaptado para Vercel Serverless Functions (<40 MB total bundle).
  Carga el motor de inferencia ligero en NumPy (modelo_edupredict_light.json)
  y expone los endpoints de predicción y métricas, además del Dashboard interactivo.
===============================================================================
"""

import json
import os
import sys
from pathlib import Path

# Fijar el directorio de trabajo a la raíz del proyecto para encontrar artefactos
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_PROJECT_ROOT)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse # noqa: E402
from pydantic import BaseModel, Field # noqa: E402
import pandas as pd # noqa: E402
import numpy as np # noqa: E402

import inferencia # noqa: E402

app = FastAPI(
    title="EduPredict Trujillo API",
    description="Sistema predictivo de deserción universitaria (Vercel Serverless)",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Carga de artefactos
# -----------------------------------------------------------------------------
_MODELO = None
_PREPROCESADOR = None
_METRICAS = None

def get_artefactos():
    global _MODELO, _PREPROCESADOR
    if _MODELO is None or _PREPROCESADOR is None:
        _MODELO, _PREPROCESADOR = inferencia.cargar_artefactos_ligeros()
    return _MODELO, _PREPROCESADOR

def get_metricas():
    global _METRICAS
    if _METRICAS is None:
        ruta = _PROJECT_ROOT / "metricas_edupredict.json"
        if ruta.exists():
            with open(ruta, "r", encoding="utf-8") as f:
                _METRICAS = json.load(f)
        else:
            _METRICAS = {}
    return _METRICAS

# Mapeos de ocupación estimada por nivel educativo (modas del dataset UCI)
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

# -----------------------------------------------------------------------------
# Esquemas Pydantic
# -----------------------------------------------------------------------------
class StudentInput(BaseModel):
    edad: int = Field(18, ge=15, le=70)
    sexo: int = Field(1, description="1: Masculino, 0: Femenino")
    carrera_codigo: int = Field(9119, description="Código UCI de la carrera equivalencia peruana")
    turno: int = Field(1, description="1: Diurno, 0: Vespertino")
    nota_admision: float = Field(14.0, ge=0.0, le=20.0)
    
    # 1er Ciclo
    matriculados_c1: int = Field(5, ge=1, le=12)
    aprobados_c1: int = Field(4, ge=0, le=12)
    nota_c1: float = Field(13.5, ge=0.0, le=20.0)
    
    # 2do Ciclo
    matriculados_c2: int = Field(5, ge=1, le=12)
    aprobados_c2: int = Field(4, ge=0, le=12)
    nota_c2: float = Field(13.0, ge=0.0, le=20.0)
    
    # Contexto Familiar y Económico
    nivel_madre: int = Field(1, description="Código de nivel educativo madre")
    nivel_padre: int = Field(1, description="Código de nivel educativo padre")
    ingreso_rmv: int = Field(1, description="0: <=1 RMV, 1: 1-2 RMV, 2: >2 RMV")
    beca: int = Field(0, description="0: No, 1: Sí")
    deudor: int = Field(0, description="0: No (pensiones al día), 1: Sí (pensiones adeudadas)")
    formacion_previa: int = Field(1, description="1: Secundaria, 42: Instituto, 2: Universidad previa")
    desplazado: int = Field(0, description="0: No, 1: Sí")


# -----------------------------------------------------------------------------
# Endpoints de API
# -----------------------------------------------------------------------------
@app.get("/api/health")
@app.get("/health")
def health_check():
    return {"status": "ok", "app": "EduPredict Trujillo API", "mode": "lightweight-numpy", "author": "Renzo Ocupa"}

@app.get("/api/metrics")
@app.get("/metrics")
def get_metrics_endpoint():
    metricas = get_metricas()
    return JSONResponse(content=metricas)

@app.post("/api/predict")
@app.post("/predict")
def predict_student(data: StudentInput):
    try:
        modelo, preprocesador = get_artefactos()
        
        # Mapeo de contexto macro económico (RMV Perú -> UCI Macro)
        macro_map = {
            0: {"unemployment": 13.0, "inflation": 3.5, "gdp": 1.5},
            1: {"unemployment": 10.0, "inflation": 3.1, "gdp": 2.5},
            2: {"unemployment": 7.0, "inflation": 2.7, "gdp": 3.8}
        }
        macro = macro_map.get(data.ingreso_rmv, macro_map[1])
        
        # Ajuste de carrera vespertina si aplica
        carrera = data.carrera_codigo
        if data.turno == 0:
            if carrera == 9238: carrera = 8014
            elif carrera == 9147: carrera = 9991
            
        # Ocupación estimada por nivel educativo
        ocu_m = OCU_MADRE_POR_NIVEL.get(data.nivel_madre, 9)
        ocu_p = OCU_PADRE_POR_NIVEL.get(data.nivel_padre, 9)
        
        # Construcción de la fila de 36 características
        fila = {
            # 18 numéricas
            "Previous qualification (grade)": round(data.nota_admision * 10.0, 1),
            "Admission grade": round(data.nota_admision * 10.0, 1),
            "Age at enrollment": data.edad,
            "Curricular units 1st sem (credited)": 0,
            "Curricular units 1st sem (enrolled)": data.matriculados_c1,
            "Curricular units 1st sem (evaluations)": data.matriculados_c1,
            "Curricular units 1st sem (approved)": data.aprobados_c1,
            "Curricular units 1st sem (grade)": float(data.nota_c1),
            "Curricular units 1st sem (without evaluations)": 0,
            "Curricular units 2nd sem (credited)": 0,
            "Curricular units 2nd sem (enrolled)": data.matriculados_c2,
            "Curricular units 2nd sem (evaluations)": data.matriculados_c2,
            "Curricular units 2nd sem (approved)": data.aprobados_c2,
            "Curricular units 2nd sem (grade)": float(data.nota_c2),
            "Curricular units 2nd sem (without evaluations)": 0,
            "Unemployment rate": macro["unemployment"],
            "Inflation rate": macro["inflation"],
            "GDP": macro["gdp"],
            
            # 18 categóricas
            "Marital status": 1,
            "Application mode": 1,
            "Application order": 1,
            "Daytime/evening attendance": data.turno,
            "Previous qualification": data.formacion_previa,
            "Nacionality": 1,
            "Mother's qualification": data.nivel_madre,
            "Father's qualification": data.nivel_padre,
            "Mother's occupation": ocu_m,
            "Father's occupation": ocu_p,
            "Displaced": data.desplazado,
            "Educational special needs": 0,
            "Debtor": data.deudor,
            "Tuition fees up to date": 1 if data.deudor == 0 else 0,
            "Gender": data.sexo,
            "Scholarship holder": data.beca,
            "International": 0,
            "Course": carrera
        }
        
        df = pd.DataFrame([fila])
        
        # Transformación con preprocesador ligero (StandardScaler + OneHotEncoder en NumPy)
        X = preprocesador.transform(df)
        
        # Inferencia con modelo ligero (booster float32 en NumPy puro)
        proba = modelo.predict_proba(X)[0]  # [P(Permanencia), P(Deserción)]
        prob_desercion = float(proba[1]) * 100.0
        prob_exito = float(proba[0]) * 100.0
        
        # Clasificación de nivel de riesgo
        if prob_desercion < 30.0:
            nivel_riesgo = "BAJO"
            color_riesgo = "#34a853"
            mensaje_riesgo = "Riesgo de deserción controlado. Se recomienda tutoría académica estándar."
        elif prob_desercion <= 60.0:
            nivel_riesgo = "MEDIO"
            color_riesgo = "#fbbc04"
            mensaje_riesgo = "Riesgo moderado. Se sugiere monitoreo de notas y orientación psicopedagógica."
        else:
            nivel_riesgo = "ALTO"
            color_riesgo = "#ea4335"
            mensaje_riesgo = "¡ALERTA DE RIESGO ELEVADO! Requiere intervención prioritaria del comité de tutoría."

        # Recomendaciones personalizadas
        recomendaciones = []
        if data.aprobados_c1 < data.matriculados_c1 or data.aprobados_c2 < data.matriculados_c2:
            recomendaciones.append("Acompañamiento académico reforzado en asignaturas desaprobadas.")
        if data.nota_c2 < 11.0 or data.nota_c1 < 11.0:
            recomendaciones.append("Tutoría intensiva para nivelación de competencias académicas.")
        if data.deudor == 1:
            recomendaciones.append("Evaluación con la oficina de Bienestar Universitario para convenio de pagos.")
        if data.beca == 0 and data.ingreso_rmv == 0:
            recomendaciones.append("Postulación prioritaria al programa de becas y subvenciones socioeconómicas.")
        if not recomendaciones:
            recomendaciones.append("Continuar con el plan de estudios habitual y tutoría preventiva mensual.")

        return {
            "success": True,
            "probabilidad_desercion": round(prob_desercion, 2),
            "probabilidad_exito": round(prob_exito, 2),
            "nivel_riesgo": nivel_riesgo,
            "color_riesgo": color_riesgo,
            "mensaje_riesgo": mensaje_riesgo,
            "recomendaciones": recomendaciones,
            "transformacion_peru": {
                "nota_admision_uci": round(data.nota_admision * 10.0, 1),
                "unemployment_rate": macro["unemployment"],
                "inflation_rate": macro["inflation"],
                "gdp": macro["gdp"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Dashboard Web UI 100% Responsivo Móvil + Animado (HTML5 + CSS + Plotly.js)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
@app.get("/index.py", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
def serve_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
    <title>EduPredict Trujillo - Prototipo Desarrollado por Renzo Ocupa</title>

    <!-- Google Fonts & Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined">

    <!-- Plotly.js para gráficos client-side interactivos -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

    <style>
        :root {
            --primary: #4285F4;
            --primary-gradient: linear-gradient(135deg, #4285F4 0%, #1A73E8 100%);
            --primary-hover: #3367D6;
            --success: #34A853;
            --warning: #FBBC04;
            --danger: #EA4335;
            --bg-color: #F8F9FA;
            --surface: #FFFFFF;
            --text-main: #202124;
            --text-sub: #5F6368;
            --border-color: #E0E0E0;
            --shadow-sm: 0 2px 6px rgba(60,64,67,0.08);
            --shadow-hover: 0 8px 24px rgba(60,64,67,0.16);
            --radius-md: 14px;
            --radius-lg: 18px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        html, body {
            width: 100%;
            overflow-x: hidden;
        }

        body {
            font-family: 'Roboto', 'Google Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.5;
            padding-bottom: 3rem;
        }

        /* App Bar Header responsivo */
        .app-bar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 0.9rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: var(--shadow-sm);
            position: sticky;
            top: 0;
            z-index: 100;
            animation: fadeInDown 0.6s ease;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .brand-icon-box {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: var(--primary-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            box-shadow: 0 4px 12px rgba(66, 133, 244, 0.3);
            flex-shrink: 0;
        }

        .brand-title {
            font-family: 'Google Sans', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.3px;
        }

        .brand-subtitle {
            font-size: 0.8rem;
            color: var(--text-sub);
        }

        .author-badge {
            background: #E8F0FE;
            color: #1967D2;
            padding: 0.4rem 0.9rem;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            border: 1px solid #D2E3FC;
            transition: all 0.3s ease;
            white-space: nowrap;
        }

        .container {
            max-width: 1400px;
            margin: 1.5rem auto;
            padding: 0 1.2rem;
            animation: fadeInUp 0.7s ease;
        }

        /* Tabs optimizados para móvil con scroll táctil suave */
        .tabs-wrapper {
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none; /* Firefox */
            margin-bottom: 1.5rem;
        }

        .tabs-wrapper::-webkit-scrollbar { display: none; } /* Chrome/Safari */

        .tabs {
            display: flex;
            gap: 0.5rem;
            border-bottom: 2px solid var(--border-color);
            min-width: max-content;
        }

        .tab-btn {
            padding: 0.85rem 1.2rem;
            border: none;
            background: none;
            font-family: 'Google Sans', sans-serif;
            font-size: 0.92rem;
            font-weight: 500;
            color: var(--text-sub);
            cursor: pointer;
            border-bottom: 3px solid transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
            border-radius: 8px 8px 0 0;
            white-space: nowrap;
        }

        .tab-btn:hover {
            color: var(--primary);
            background-color: rgba(66, 133, 244, 0.05);
        }

        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
            font-weight: 700;
            background-color: rgba(66, 133, 244, 0.08);
        }

        .tab-content {
            display: none;
            opacity: 0;
            transform: translateY(10px);
            transition: opacity 0.4s ease, transform 0.4s ease;
        }

        .tab-content.active {
            display: block;
            opacity: 1;
            transform: translateY(0);
        }

        /* Layout Grid Responsivo */
        .grid-layout, .grid-2col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        /* Cards Animadas */
        .card {
            background: var(--surface);
            border-radius: var(--radius-md);
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            margin-bottom: 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .card:hover {
            box-shadow: var(--shadow-hover);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-family: 'Google Sans', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1.2rem;
            color: var(--text-main);
            border-bottom: 1px solid #F1F3F4;
            padding-bottom: 0.6rem;
        }

        .card-header .material-icons-outlined { color: var(--primary); }

        /* Formulario 100% táctil para celular */
        .form-section { margin-bottom: 1.2rem; }

        .form-section-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--primary);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 0.8rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-main);
        }

        /* Tamaño 16px evita zoom automático molesto en iOS Safari */
        input, select {
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            background-color: #FAFAFA;
            transition: all 0.25s ease;
            width: 100%;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(66, 133, 244, 0.15);
            background-color: #FFFFFF;
        }

        .btn-predict {
            width: 100%;
            padding: 1rem;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 10px;
            font-family: 'Google Sans', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            box-shadow: 0 4px 14px rgba(66, 133, 244, 0.35);
            transition: all 0.3s ease;
            margin-top: 1rem;
            touch-action: manipulation;
        }

        .btn-predict:active { transform: scale(0.98); }

        /* Resultados */
        .result-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 1.5rem;
            border-radius: var(--radius-md);
            background-color: #FFFFFF;
            border: 2px solid var(--border-color);
            transition: border-color 0.4s ease;
        }

        .risk-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 1.4rem;
            border-radius: 30px;
            font-family: 'Google Sans', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            color: white;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
        }

        .gauge-container {
            width: 100%;
            height: 240px;
        }

        .rec-box {
            background-color: #F8F9FA;
            border-left: 5px solid var(--primary);
            padding: 1.1rem;
            border-radius: 10px;
            width: 100%;
            text-align: left;
            margin-top: 1rem;
            animation: fadeIn 0.5s ease;
        }

        .rec-box h4 {
            font-family: 'Google Sans', sans-serif;
            font-size: 0.95rem;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }

        .rec-list {
            padding-left: 1.2rem;
            font-size: 0.88rem;
            color: var(--text-sub);
        }

        .rec-list li { margin-bottom: 0.35rem; }

        /* Cards de Métricas */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: white;
            padding: 1.2rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-color);
            text-align: center;
            box-shadow: var(--shadow-sm);
        }

        .metric-val {
            font-family: 'Google Sans', sans-serif;
            font-size: 1.9rem;
            font-weight: 700;
            color: var(--primary);
        }

        .metric-lbl {
            font-size: 0.82rem;
            color: var(--text-sub);
            margin-top: 0.2rem;
        }

        /* Tabla responsiva para móvil */
        .table-responsive {
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border-radius: 8px;
        }

        .custom-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 0.5rem;
            min-width: 400px;
        }

        .custom-table th, .custom-table td {
            padding: 0.75rem 0.8rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .custom-table th {
            background-color: #F1F3F4;
            font-weight: 700;
            color: var(--text-main);
        }

        .plot-container-md {
            width: 100%;
            height: 360px;
        }

        /* Footer del creador */
        .creator-footer {
            text-align: center;
            padding: 1.5rem 0;
            margin-top: 2rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-sub);
            font-size: 0.85rem;
        }

        .creator-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: #FFFFFF;
            padding: 0.5rem 1.1rem;
            border-radius: 30px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            font-weight: 500;
            color: var(--text-main);
        }

        /* Keyframe Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* MEDIA QUERIES ESPECÍFICAS PARA CELULAR */
        @media (max-width: 768px) {
            .app-bar {
                padding: 0.8rem 1rem;
                flex-direction: column;
                align-items: flex-start;
                gap: 0.6rem;
            }

            .author-badge {
                width: 100%;
                justify-content: center;
                font-size: 0.78rem;
                padding: 0.35rem 0.6rem;
            }

            .container {
                padding: 0 0.8rem;
                margin: 1rem auto;
            }

            .grid-layout, .grid-2col {
                grid-template-columns: 1fr;
                gap: 1rem;
            }

            .card {
                padding: 1.1rem;
                margin-bottom: 1rem;
                border-radius: 12px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 0.7rem;
            }

            .metric-card {
                padding: 0.9rem;
            }

            .metric-val {
                font-size: 1.5rem;
            }

            .plot-container-md {
                height: 300px;
            }
        }

        @media (max-width: 480px) {
            .brand-title { font-size: 1.15rem; }
            .metrics-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header class="app-bar">
        <div class="brand-container">
            <div class="brand-icon-box">
                <span class="material-icons-outlined">school</span>
            </div>
            <div>
                <div class="brand-title">EduPredict Trujillo</div>
                <div class="brand-subtitle">Sistema Predictivo de Deserción Universitaria | UCV (2026)</div>
            </div>
        </div>
        <div class="author-badge">
            <span class="material-icons-outlined" style="font-size:1rem;">code</span>
            Desarrollado por Renzo Ocupa (17/09/2026)
        </div>
    </header>

    <div class="container">
        <!-- Navigation Tabs Responsivos -->
        <div class="tabs-wrapper">
            <nav class="tabs">
                <button class="tab-btn active" onclick="switchTab('tab-predict')">
                    <span class="material-icons-outlined">psychology</span> Diagnóstico Predictivo
                </button>
                <button class="tab-btn" onclick="switchTab('tab-metrics')">
                    <span class="material-icons-outlined">bar_chart</span> Métricas & Análisis
                </button>
                <button class="tab-btn" onclick="switchTab('tab-info')">
                    <span class="material-icons-outlined">info</span> Metodología Perú
                </button>
            </nav>
        </div>

        <!-- TAB 1: DIAGNÓSTICO PREDICTIVO -->
        <div id="tab-predict" class="tab-content active">
            <div class="grid-layout">
                <!-- FORMULARIO -->
                <div class="card">
                    <div class="card-header">
                        <span class="material-icons-outlined">assignment</span>
                        Formulario del Estudiante (Escala Perú 0-20)
                    </div>

                    <form id="predictionForm" onsubmit="handlePredict(event)">
                        <!-- Sección 1 -->
                        <div class="form-section">
                            <div class="form-section-title">1. Datos Personales y Carrera</div>
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="edad">Edad al Matricularse:</label>
                                    <input type="number" id="edad" value="18" min="15" max="70" required>
                                </div>
                                <div class="form-group">
                                    <label for="sexo">Sexo:</label>
                                    <select id="sexo">
                                        <option value="1">Masculino</option>
                                        <option value="0">Femenino</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="carrera_codigo">Carrera (Equivalencia Peruana):</label>
                                    <select id="carrera_codigo">
                                        <option value="9119" selected>Ingeniería de Sistemas e Informática</option>
                                        <option value="9147">Administración de Empresas</option>
                                        <option value="9500">Enfermería</option>
                                        <option value="9254">Turismo y Hotelería</option>
                                        <option value="9003">Agronomía / Ing. Agrónoma</option>
                                        <option value="9773">Comunicación y Periodismo</option>
                                        <option value="9853">Educación (Docencia)</option>
                                        <option value="9238">Trabajo Social</option>
                                        <option value="9070">Diseño Gráfico / Comunicación</option>
                                        <option value="9670">Marketing y Publicidad</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="turno">Turno de Estudio:</label>
                                    <select id="turno">
                                        <option value="1" selected>Diurno</option>
                                        <option value="0">Vespertino / Nocturno</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="nota_admision">Nota de Admisión (0 - 20):</label>
                                    <input type="number" id="nota_admision" value="14.0" min="0" max="20" step="0.5" required>
                                </div>
                            </div>
                        </div>

                        <!-- Sección 2 -->
                        <div class="form-section">
                            <div class="form-section-title">2. Rendimiento Académico en la Universidad</div>
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="matriculados_c1">Cursos Matriculados (Ciclo 1):</label>
                                    <input type="number" id="matriculados_c1" value="5" min="1" max="10" required>
                                </div>
                                <div class="form-group">
                                    <label for="aprobados_c1">Cursos Aprobados (Ciclo 1):</label>
                                    <input type="number" id="aprobados_c1" value="4" min="0" max="10" required>
                                </div>
                                <div class="form-group">
                                    <label for="nota_c1">Nota Promedio Ciclo 1 (0 - 20):</label>
                                    <input type="number" id="nota_c1" value="13.5" min="0" max="20" step="0.1" required>
                                </div>
                                <div class="form-group">
                                    <label for="matriculados_c2">Cursos Matriculados (Ciclo 2):</label>
                                    <input type="number" id="matriculados_c2" value="5" min="1" max="10" required>
                                </div>
                                <div class="form-group">
                                    <label for="aprobados_c2">Cursos Aprobados (Ciclo 2):</label>
                                    <input type="number" id="aprobados_c2" value="4" min="0" max="10" required>
                                </div>
                                <div class="form-group">
                                    <label for="nota_c2">Nota Promedio Ciclo 2 (0 - 20):</label>
                                    <input type="number" id="nota_c2" value="13.0" min="0" max="20" step="0.1" required>
                                </div>
                            </div>
                        </div>

                        <!-- Sección 3 -->
                        <div class="form-section">
                            <div class="form-section-title">3. Contexto Socioeconómico y Familiar</div>
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="ingreso_rmv">Ingreso Familiar (RMV S/ 1,130):</label>
                                    <select id="ingreso_rmv">
                                        <option value="0">Hasta 1 RMV (≤ S/ 1,130)</option>
                                        <option value="1" selected>1 a 2 RMV (S/ 1,130 – S/ 2,260)</option>
                                        <option value="2">Más de 2 RMV (> S/ 2,260)</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="deudor">¿Pensiones al día?:</label>
                                    <select id="deudor">
                                        <option value="0" selected>Sí, pensiones al día</option>
                                        <option value="1">No, presenta deuda de pensiones</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="beca">¿Cuenta con Beca?:</label>
                                    <select id="beca">
                                        <option value="0" selected>No tiene beca</option>
                                        <option value="1">Sí, cuenta con beca de estudios</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="nivel_madre">Estudios de la Madre:</label>
                                    <select id="nivel_madre">
                                        <option value="1" selected>Secundaria completa</option>
                                        <option value="37">Primaria completa</option>
                                        <option value="42">Instituto / Técnica</option>
                                        <option value="3">Universidad completa</option>
                                        <option value="35">Sin estudios</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="nivel_padre">Estudios del Padre:</label>
                                    <select id="nivel_padre">
                                        <option value="1" selected>Secundaria completa</option>
                                        <option value="37">Primaria completa</option>
                                        <option value="42">Instituto / Técnica</option>
                                        <option value="3">Universidad completa</option>
                                        <option value="35">Sin estudios</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <button type="submit" class="btn-predict">
                            <span class="material-icons-outlined">analytics</span>
                            Calcular Probabilidad de Deserción
                        </button>
                    </form>
                </div>

                <!-- RESULTADOS -->
                <div class="card">
                    <div class="card-header">
                        <span class="material-icons-outlined">assessment</span>
                        Resultado del Diagnóstico Preventivo
                    </div>

                    <div id="resultsContainer" class="result-card">
                        <div id="riskBadge" class="risk-badge" style="background-color: var(--primary);">
                            <span class="material-icons-outlined">hourglass_empty</span> Realice una predicción
                        </div>

                        <!-- Gauge Chart -->
                        <div id="gaugePlot" class="gauge-container"></div>

                        <div id="recBox" class="rec-box" style="display: none;">
                            <h4>Plan de Acción Recomendado para el Tutor:</h4>
                            <ul id="recList" class="rec-list"></ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: MÉTRICAS DEL MODELO & ANÁLISIS DE DATOS -->
        <div id="tab-metrics" class="tab-content">
            <!-- Métricas KPI Cards -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-val">88.25%</div>
                    <div class="metric-lbl">Accuracy (Exactitud Global)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">84.88%</div>
                    <div class="metric-lbl">Precision (Deserción)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">77.11%</div>
                    <div class="metric-lbl">Recall (Deserción)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">80.81%</div>
                    <div class="metric-lbl">F1-Score (Deserción)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">93.61%</div>
                    <div class="metric-lbl">ROC AUC (Discriminación)</div>
                </div>
            </div>

            <!-- Gráficos de Métricas y Matriz de Confusión -->
            <div class="grid-2col">
                <div class="card">
                    <div class="card-header">
                        <span class="material-icons-outlined">grid_on</span>
                        Matriz de Confusión (Evaluación sobre Test original)
                    </div>
                    <div id="confusionPlot" class="plot-container-md"></div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span class="material-icons-outlined">leaderboard</span>
                        Top Variables más Influyentes (XGBoost Feature Importance)
                    </div>
                    <div id="importancePlot" class="plot-container-md"></div>
                </div>
            </div>

            <!-- Tabla de Resumen de Clases y Datos del Dataset -->
            <div class="grid-2col">
                <div class="card">
                    <div class="card-header">
                        <span class="material-icons-outlined">pie_chart</span>
                        Distribución de Clases del Dataset UCI (4,424 Filas)
                    </div>
                    <div class="table-responsive">
                        <table class="custom-table">
                            <thead>
                                <tr>
                                    <th>Estado Final</th>
                                    <th>Clase Binarizada</th>
                                    <th>Estudiantes</th>
                                    <th>Porcentaje</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Graduate (Graduado)</strong></td>
                                    <td>0 (Éxito)</td>
                                    <td>2,209</td>
                                    <td>49.9%</td>
                                </tr>
                                <tr>
                                    <td><strong>Dropout (Abandonó)</strong></td>
                                    <td>1 (Riesgo)</td>
                                    <td>1,421</td>
                                    <td>32.1%</td>
                                </tr>
                                <tr>
                                    <td><strong>Enrolled (Continuante)</strong></td>
                                    <td>0 (Éxito)</td>
                                    <td>794</td>
                                    <td>17.9%</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span class="material-icons-outlined">table_chart</span>
                        Matriz de Correlaciones Académicas
                    </div>
                    <div id="correlationPlot" class="plot-container-md"></div>
                </div>
            </div>
        </div>

        <!-- TAB 3: METODOLOGÍA & ADAPTACIÓN PERÚ -->
        <div id="tab-info" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="material-icons-outlined">gavel</span>
                    Declaración Metodológica y Adaptación al Contexto de Trujillo (Perú)
                </div>
                <div style="font-size: 0.95rem; color: var(--text-main); line-height: 1.7;">
                    <p style="margin-bottom: 1rem;">
                        <strong>1. Naturaleza de la Adaptación:</strong> El modelo de Machine Learning utiliza el dataset oficial <em>"Predict students' dropout and academic success"</em> del UCI Machine Learning Repository. Este sistema representa una propuesta de aplicación al contexto universitario de Trujillo (UCV) para asistencia preventiva de tutores.
                    </p>
                    <p style="margin-bottom: 1rem;">
                        <strong>2. Transformaciones Transparentes:</strong>
                        <br>• <em>Notas (0-20 peruana):</em> Se traducen a la escala original del dataset (0-200) multiplicando por 10.
                        <br>• <em>Contexto Económico:</em> El ingreso familiar basado en la Remuneración Mínima Vital (RMV S/ 1,130) se asigna transparentemente a los parámetros de contexto macroeconómico del modelo.
                    </p>
                    <p>
                        <strong>3. Modelo Ligero para Vercel:</strong> El sistema de inferencia ligero reduce el bundle de la aplicación de 881 MB a menos de 40 MB en NumPy puro, eliminando latencias y respetando los límites serverless de Vercel.
                    </p>
                </div>
            </div>
        </div>

        <!-- Pie de Autoría -->
        <footer class="creator-footer">
            <div class="creator-tag">
                <span class="material-icons-outlined" style="color:var(--primary);">verified</span>
                Programa desarrollado por Renzo Ocupa | 17/09/2026 Prototipo TIF Nivel III
            </div>
        </footer>
    </div>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            event.currentTarget.classList.add('active');
            const target = document.getElementById(tabId);
            target.classList.add('active');

            if (tabId === 'tab-metrics') {
                setTimeout(renderMetricsCharts, 100);
            }
        }

        // Render Gauge con ajuste responsivo de tamaño
        function renderGauge(value, title, color) {
            const isMobile = window.innerWidth <= 768;
            const data = [{
                type: "indicator",
                mode: "gauge+number",
                value: value,
                number: { suffix: "%", font: { size: isMobile ? 30 : 38, family: "Google Sans" } },
                title: { text: "Probabilidad de Deserción", font: { size: isMobile ? 13 : 15, family: "Google Sans" } },
                gauge: {
                    axis: { range: [0, 100], tickwidth: 1, tickcolor: "#5F6368" },
                    bar: { color: color, width: 0.25 },
                    bgcolor: "white",
                    borderwidth: 1,
                    bordercolor: "#E0E0E0",
                    steps: [
                        { range: [0, 30], color: "rgba(52, 168, 83, 0.15)" },
                        { range: [30, 60], color: "rgba(251, 188, 4, 0.15)" },
                        { range: [60, 100], color: "rgba(234, 67, 53, 0.15)" }
                    ]
                }
            }];

            const layout = {
                margin: { t: 25, r: 20, l: 20, b: 25 },
                paper_bgcolor: "rgba(0,0,0,0)",
                font: { color: "#202124", family: "Roboto" }
            };

            Plotly.react('gaugePlot', data, layout, { responsive: true, displayModeBar: false });
        }

        // Render Gráficos de Métricas de Evaluación
        function renderMetricsCharts() {
            const isMobile = window.innerWidth <= 768;
            
            // Matriz de Confusión
            const cmData = [{
                z: [[562, 39], [65, 219]],
                x: [isMobile ? 'Graduado' : 'Pred: Graduado', isMobile ? 'Dropout' : 'Pred: Dropout'],
                y: [isMobile ? 'Graduado' : 'Real: Graduado', isMobile ? 'Dropout' : 'Real: Dropout'],
                type: 'heatmap',
                colorscale: 'Blues',
                showscale: false
            }];
            const cmLayout = {
                margin: { t: 20, r: 10, l: isMobile ? 70 : 110, b: 40 },
                annotations: [
                    { x: isMobile ? 'Graduado' : 'Pred: Graduado', y: isMobile ? 'Graduado' : 'Real: Graduado', text: '562 (TN)', showarrow: false, font: { size: isMobile ? 12 : 15, color: 'white' } },
                    { x: isMobile ? 'Dropout' : 'Pred: Dropout', y: isMobile ? 'Graduado' : 'Real: Graduado', text: '39 (FP)', showarrow: false, font: { size: isMobile ? 12 : 15, color: 'black' } },
                    { x: isMobile ? 'Graduado' : 'Pred: Graduado', y: isMobile ? 'Dropout' : 'Real: Dropout', text: '65 (FN)', showarrow: false, font: { size: isMobile ? 12 : 15, color: 'black' } },
                    { x: isMobile ? 'Dropout' : 'Pred: Dropout', y: isMobile ? 'Dropout' : 'Real: Dropout', text: '219 (TP)', showarrow: false, font: { size: isMobile ? 12 : 15, color: 'white' } }
                ]
            };
            Plotly.react('confusionPlot', cmData, cmLayout, { responsive: true, displayModeBar: false });

            // Importance Chart
            const impData = [{
                x: [0.1065, 0.0723, 0.0645, 0.0317, 0.0196, 0.0195, 0.0182, 0.0165, 0.0163],
                y: [
                    'Aprobados Ciclo 2',
                    'Pensiones al día',
                    'Estudios Padre',
                    'Carrera Cursada',
                    'Aprobados Ciclo 1',
                    'Matriculados Ciclo 1',
                    'Posee Beca',
                    'Género',
                    'Formación Previa'
                ].reverse(),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#4285F4' }
            }];
            const impLayout = {
                margin: { t: 15, r: 15, l: isMobile ? 110 : 130, b: 30 },
                xaxis: { title: 'Importancia Relativa' }
            };
            Plotly.react('importancePlot', impData, impLayout, { responsive: true, displayModeBar: false });

            // Correlation Chart
            const corrData = [{
                z: [
                    [1.0, 0.58, 0.35, 0.42, 0.12],
                    [0.58, 1.0, 0.28, 0.39, 0.08],
                    [0.35, 0.28, 1.0, 0.76, -0.05],
                    [0.42, 0.39, 0.76, 1.0, -0.09],
                    [0.12, 0.08, -0.05, -0.09, 1.0]
                ],
                x: ['Admisión', 'Previa', 'Aprob C1', 'Aprob C2', 'Macro'],
                y: ['Admisión', 'Previa', 'Aprob C1', 'Aprob C2', 'Macro'],
                type: 'heatmap',
                colorscale: 'RdBu',
                zmin: -1, zmax: 1
            }];
            const corrLayout = {
                margin: { t: 15, r: 15, l: 65, b: 50 }
            };
            Plotly.react('correlationPlot', corrData, corrLayout, { responsive: true, displayModeBar: false });
        }

        // Carga inicial y redimensionamiento dinámico
        window.addEventListener('DOMContentLoaded', () => {
            renderGauge(15.5, "Diagnóstico Inicial", "#34A853");
        });

        window.addEventListener('resize', () => {
            renderGauge(15.5, "Diagnóstico Inicial", "#34A853");
            renderMetricsCharts();
        });

        async function handlePredict(e) {
            e.preventDefault();
            
            const payload = {
                edad: parseInt(document.getElementById('edad').value),
                sexo: parseInt(document.getElementById('sexo').value),
                carrera_codigo: parseInt(document.getElementById('carrera_codigo').value),
                turno: parseInt(document.getElementById('turno').value),
                nota_admision: parseFloat(document.getElementById('nota_admision').value),
                matriculados_c1: parseInt(document.getElementById('matriculados_c1').value),
                aprobados_c1: parseInt(document.getElementById('aprobados_c1').value),
                nota_c1: parseFloat(document.getElementById('nota_c1').value),
                matriculados_c2: parseInt(document.getElementById('matriculados_c2').value),
                aprobados_c2: parseInt(document.getElementById('aprobados_c2').value),
                nota_c2: parseFloat(document.getElementById('nota_c2').value),
                ingreso_rmv: parseInt(document.getElementById('ingreso_rmv').value),
                deudor: parseInt(document.getElementById('deudor').value),
                beca: parseInt(document.getElementById('beca').value),
                nivel_madre: parseInt(document.getElementById('nivel_madre').value),
                nivel_padre: parseInt(document.getElementById('nivel_padre').value),
                formacion_previa: 1,
                desplazado: 0
            };

            try {
                let endpoint = '/api/predict';
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                }).catch(async () => {
                    return await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                });

                if (!res.ok) throw new Error("Error en la predicción serverless");

                const result = await res.json();
                
                // Actualizar Badge de Riesgo con animación
                const badge = document.getElementById('riskBadge');
                badge.style.backgroundColor = result.color_riesgo;
                badge.innerHTML = `<span class="material-icons-outlined">warning</span> RIESGO ${result.nivel_riesgo} (${result.probabilidad_desercion}%)`;

                // Render Gauge Plotly
                renderGauge(result.probabilidad_desercion, result.nivel_riesgo, result.color_riesgo);

                // Mostrar recomendaciones
                const recBox = document.getElementById('recBox');
                const recList = document.getElementById('recList');
                recList.innerHTML = result.recomendaciones.map(r => `<li>${r}</li>`).join('');
                recBox.style.display = 'block';

                // Scroll suave hasta el resultado en celular
                if (window.innerWidth <= 768) {
                    document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });
                }

            } catch (err) {
                alert("Error al realizar la predicción: " + err.message);
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

# Catch-all GET handler para asegurar que cualquier ruta sirva la Dashboard UI
@app.api_route("/{path_name:path}", methods=["GET"])
def catch_all_dashboard(request: Request, path_name: str):
    if path_name.startswith("api/metrics") or path_name == "metrics":
        return get_metrics_endpoint()
    if path_name.startswith("api/health") or path_name == "health":
        return health_check()
    return serve_dashboard()