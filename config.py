# =============================================================================
# config.py — Configuración central del pipeline de predicción de parkings
# Todos los parámetros modificables están aquí. No tocar los otros scripts
# para ajustes rutinarios.
# =============================================================================

from pathlib import Path

# -----------------------------------------------------------------------------
# RUTAS DE DATOS
# -----------------------------------------------------------------------------
DATA_DIR  = Path(".")      # carpeta con los 3 CSVs de entrada (mismo directorio)
OUT_DIR   = Path("outputs")  # carpeta de salida (modelos, métricas, predicciones)

PARKING_CSV  = DATA_DIR / "parking_data.csv"
METEO_CSV    = DATA_DIR / "meteo_data.csv"
FESTIVOS_CSV = DATA_DIR / "festivos.csv"

# -----------------------------------------------------------------------------
# PARKINGS
# -----------------------------------------------------------------------------
# Parkings a excluir antes de cualquier procesado (datos inutilizables)
PARKINGS_EXCLUIR = ["LI", "SA"]  # EL LIMONAR: 100% nulls en target

# Override manual de alpha por parking (si un parking sale mal, ajustar aquí)
# AN: alta varianza + picos bimodales → alpha más agresivo para empujar predicción
# PA: patrón estacional fuerte (verano+fds) subestimado con 0.70
ALPHA_POR_PARKING: dict = {
    "AN": 0.90,   # mejor recall t30/t60 en experimento A
    "PA": 0.75,   # mejor balance t30 sin penalizar t15
}

# -----------------------------------------------------------------------------
# TARGET Y COLUMNAS
# -----------------------------------------------------------------------------
TARGET_COL    = "ocupadas_oficial"   # variable a predecir (plazas absolutas)
CAPACITY_COL  = "plazas_totales_oficial"
LIBRES_COL    = "libres"
TIMESTAMP_COL = "timestamp"
PARKING_ID    = "parking_id"
PARKING_NAME  = "parking_name"

# Columnas a eliminar del CSV de parking (no son features ni target)
COLS_ELIMINAR_PARKING = [
    "plazas_totales_crudo",
    "ocupadas_crudo",
    "pct_ocupacion_crudo",
    "pct_ocupacion_oficial",   # derivado, nunca entra al modelo
    "source_format",
    # lat/lon se guardan en tabla de referencia, no en el modelo
]

# Columnas meteorológicas a conservar (el resto se descarta)
COLS_METEO_USAR = ["timestamp_local", "temperature_2m", "precipitation", "cloud_cover"]

# -----------------------------------------------------------------------------
# GRANULARIDAD Y FEATURES TEMPORALES
# -----------------------------------------------------------------------------
FREQ = "15min"   # granularidad de la serie temporal

# Lags en pasos de 15 minutos
# lag_1=15min, lag_2=30min, lag_4=1h, lag_8=2h, lag_96=24h
LAGS = [1, 2, 4, 8, 96]

# Ventanas de rolling (en pasos de 15min)
# 4 pasos = 1h,  12 pasos = 3h
ROLLING_WINDOWS = [4, 12]

# -----------------------------------------------------------------------------
# HORIZONTES DE PREDICCIÓN
# -----------------------------------------------------------------------------
# Clave: nombre del horizonte | Valor: pasos a futuro (1 paso = 15 min)
HORIZONS = {
    "t15": 1,   # 15 minutos
    "t30": 2,   # 30 minutos
    "t45": 3,   # 45 minutos — tiempo de movilización urbana en Málaga
    "t60": 4,   # 60 minutos
}

# -----------------------------------------------------------------------------
# SPLIT TEMPORAL (porcentaje sobre registros ordenados por tiempo)
# -----------------------------------------------------------------------------
TRAIN_PCT = 0.70
VALID_PCT = 0.15
TEST_PCT  = 0.15   # debe sumar 1.0 con los anteriores

# Mínimo de semanas que debe tener el test set para ser válido
MIN_TEST_WEEKS = 3
MIN_TEST_ROWS  = MIN_TEST_WEEKS * 7 * 24 * 4   # 3 semanas × 7 días × 24h × 4 slots

# Mínimo de registros TOTALES para que un parking tenga modelo propio
# 3 meses ≈ 90 días × 96 slots/día = 8.640 filas (después de limpieza)
MIN_ROWS_PARKING = 8_640

# -----------------------------------------------------------------------------
# MODELO XGBOOST
# -----------------------------------------------------------------------------
ALPHA_GLOBAL = 0.70   # quantile loss — penaliza más infra-predicciones
                       # rango [0,1]: 0.5=mediana, >0.5 sesga hacia arriba

XGB_PARAMS = {
    "n_estimators"       : 500,
    "learning_rate"      : 0.05,
    "max_depth"          : 6,
    "subsample"          : 0.8,
    "colsample_bytree"   : 0.8,
    "reg_lambda"         : 1.0,
    "objective"          : "reg:quantileerror",
    "tree_method"        : "hist",
    "n_jobs"             : -1,
    "random_state"       : 42,
    "early_stopping_rounds": 30,
    "eval_metric"        : "quantile",
}

# -----------------------------------------------------------------------------
# VALIDACIÓN
# -----------------------------------------------------------------------------
# Umbral de pico: % de la capacidad total del parking
# Por encima de este valor se considera que el parking está en pico
PEAK_THRESHOLD_PCT = 0.85   # 85% de capacidad

# -----------------------------------------------------------------------------
# SEMÁFORO DE SATURACIÓN
# -----------------------------------------------------------------------------
# Índice: slots_hasta_sat = espacios_libres / delta_pred_t15  (1 slot = 15 min)
# Si delta ≤ 0 (parking vaciándose o estable) → verde directo
SATURACION_UMBRAL_ROJO     = 4    # < 4 slots  (<  1 hora)  → 🔴 rojo
SATURACION_UMBRAL_AMARILLO = 8    # < 8 slots  (<  2 horas) → 🟡 amarillo
                                   # ≥ 8 slots               → 🟢 verde

# Número de features a mostrar en importancia y SHAP
TOP_N_FEATURES = 15

# Máximo de filas para el cálculo de SHAP (por rendimiento)
SHAP_SAMPLE_N = 2_000

# Tramos de ocupación para análisis de error (en % de capacidad)
# Se usan SOLO para validación, nunca como target ni feature
TRAMOS_OCUPACION = {
    "bajo"  : (0.00, 0.50),
    "medio" : (0.50, 0.85),
    "alto"  : (0.85, 1.01),
}

# -----------------------------------------------------------------------------
# MECANISMO DE OVERRIDE — usado por run_pipeline.py para experimentos
# run_pipeline.py escribe _config_override.json antes de ejecutar los pasos;
# este bloque lo aplica en tiempo de carga sin tocar los scripts 01-04.
# NO modificar a mano — editar los parámetros en las secciones de arriba.
# -----------------------------------------------------------------------------
_OVERRIDE_FILE = Path(__file__).parent / "_config_override.json"
if _OVERRIDE_FILE.exists():
    import json as _json
    _ov = _json.loads(_OVERRIDE_FILE.read_text(encoding="utf-8"))
    if "ALPHA_GLOBAL"         in _ov: ALPHA_GLOBAL        = float(_ov["ALPHA_GLOBAL"])
    if "ALPHA_POR_PARKING"    in _ov: ALPHA_POR_PARKING.update(_ov["ALPHA_POR_PARKING"])
    if "PEAK_THRESHOLD_PCT"   in _ov: PEAK_THRESHOLD_PCT  = float(_ov["PEAK_THRESHOLD_PCT"])
    if "PARKINGS_EXCLUIR"     in _ov: PARKINGS_EXCLUIR    = list(_ov["PARKINGS_EXCLUIR"])
    if "MIN_TEST_WEEKS"       in _ov: MIN_TEST_WEEKS       = int(_ov["MIN_TEST_WEEKS"])
    if "N_ESTIMATORS"         in _ov: XGB_PARAMS["n_estimators"] = int(_ov["N_ESTIMATORS"])
    if "MAX_DEPTH"            in _ov: XGB_PARAMS["max_depth"]    = int(_ov["MAX_DEPTH"])
    if "LEARNING_RATE"        in _ov: XGB_PARAMS["learning_rate"] = float(_ov["LEARNING_RATE"])
