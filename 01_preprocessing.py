# =============================================================================
# 01_preprocessing.py — Carga, limpieza, feature engineering y splits
#
# Output:
#   outputs/parking_reference.csv      → tabla estática (id, nombre, lat, lon, capacidad)
#   outputs/audit_parkings.csv         → auditoría: registros, cobertura, decisión
#   outputs/processed/<parking_id>.parquet  → datos limpios y features por parking
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    PARKING_CSV, METEO_CSV, FESTIVOS_CSV,
    OUT_DIR, PARKING_ID, PARKING_NAME, TIMESTAMP_COL,
    TARGET_COL, CAPACITY_COL, LIBRES_COL,
    COLS_ELIMINAR_PARKING, COLS_METEO_USAR,
    PARKINGS_EXCLUIR, FREQ, LAGS, ROLLING_WINDOWS, HORIZONS,
    TRAIN_PCT, VALID_PCT, TEST_PCT,
    MIN_ROWS_PARKING, MIN_TEST_ROWS,
)

PROC_DIR = OUT_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. CARGA Y LIMPIEZA — PARKING
# =============================================================================
print("=" * 70)
print("PASO 1: Carga y limpieza de datos de parking")
print("=" * 70)

df_raw = pd.read_csv(PARKING_CSV)
df_raw.columns = df_raw.columns.str.strip().str.lower()

# Normalizar nombres para compatibilidad con config.py (que usa nombres originales)
rename_map = {
    "timestamp"             : "timestamp",
    "parking_id"            : "parking_id",
    "parking_name"          : "parking_name",
    "latitud"               : "latitud",
    "longitud"              : "longitud",
    "plazas_totales_oficial": "plazas_totales_oficial",
    "libres"                : "libres",
    "ocupadas_oficial"      : "ocupadas_oficial",
    # columnas a eliminar se procesan abajo
}

df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])

# Excluir parkings sin datos útiles
df_raw = df_raw[~df_raw["parking_id"].isin(PARKINGS_EXCLUIR)].copy()
print(f"Parkings excluidos manualmente: {PARKINGS_EXCLUIR}")
print(f"Parkings restantes: {df_raw['parking_id'].nunique()}")

# Validación de integridad: libres + ocupadas == totales
check = df_raw.dropna(subset=["libres", "ocupadas_oficial", "plazas_totales_oficial"])
inconsistentes = (check["libres"] + check["ocupadas_oficial"] != check["plazas_totales_oficial"]).sum()
if inconsistentes > 0:
    print(f"⚠️  ALERTA: {inconsistentes} registros con libres+ocupadas ≠ totales → eliminando")
    mask_ok = (df_raw["libres"] + df_raw["ocupadas_oficial"] == df_raw["plazas_totales_oficial"])
    df_raw = df_raw[mask_ok | df_raw[["libres","ocupadas_oficial","plazas_totales_oficial"]].isnull().any(axis=1)].copy()
else:
    print("✅ Integridad perfecta: libres + ocupadas == totales en todos los registros")

# Eliminar columnas no necesarias
cols_drop = [c for c in COLS_ELIMINAR_PARKING if c in df_raw.columns]
df_raw.drop(columns=cols_drop, inplace=True)
print(f"Columnas eliminadas del parking: {cols_drop}")


# =============================================================================
# 2. TABLA DE REFERENCIA DE PARKINGS (estática, no entra al modelo)
# =============================================================================
print("\n" + "=" * 70)
print("PASO 2: Tabla de referencia de parkings")
print("=" * 70)

parking_ref = (
    df_raw.groupby("parking_id")
    .agg(
        parking_name      = ("parking_name", "first"),
        latitud           = ("latitud", "first"),
        longitud          = ("longitud", "first"),
        plazas_totales    = ("plazas_totales_oficial", "first"),
        primer_registro   = ("timestamp", "min"),
        ultimo_registro   = ("timestamp", "max"),
        total_registros   = ("timestamp", "count"),
    )
    .reset_index()
)

parking_ref.to_csv(OUT_DIR / "parking_reference.csv", index=False)
print("Tabla de referencia guardada → parking_reference.csv")
print(parking_ref[["parking_id","parking_name","plazas_totales","primer_registro","ultimo_registro","total_registros"]].to_string(index=False))


# =============================================================================
# 3. CARGA Y LIMPIEZA — METEOROLOGÍA
# =============================================================================
print("\n" + "=" * 70)
print("PASO 3: Carga y limpieza de meteorología")
print("=" * 70)

df_meteo = pd.read_csv(METEO_CSV, usecols=COLS_METEO_USAR)
df_meteo.columns = df_meteo.columns.str.strip().str.lower()
df_meteo["timestamp_local"] = pd.to_datetime(df_meteo["timestamp_local"])
df_meteo = df_meteo.sort_values("timestamp_local").reset_index(drop=True)

# Eliminar duplicados de timestamp (por cambio de coordenadas del API)
df_meteo = df_meteo.drop_duplicates(subset=["timestamp_local"], keep="last")

print(f"Registros meteo: {len(df_meteo):,} | Rango: {df_meteo['timestamp_local'].min()} → {df_meteo['timestamp_local'].max()}")
print(f"Columnas meteo: {df_meteo.columns.tolist()}")
print(f"Nulls: {df_meteo.isnull().sum().to_dict()}")


# =============================================================================
# 4. CARGA Y LIMPIEZA — FESTIVOS
# =============================================================================
print("\n" + "=" * 70)
print("PASO 4: Carga y limpieza de festivos")
print("=" * 70)

df_fest = pd.read_csv(FESTIVOS_CSV, usecols=["Fecha", "Festivo"])
df_fest.columns = ["fecha", "festivo"]
df_fest["fecha"]   = pd.to_datetime(df_fest["fecha"], format="%m/%d/%Y").dt.date
df_fest["festivo"] = df_fest["festivo"].astype(int)

print(f"Festivos cargados: {len(df_fest)} días | Festivos=1: {df_fest['festivo'].sum()}")


# =============================================================================
# 5. PROCESADO POR PARKING (bucle principal)
# =============================================================================
print("\n" + "=" * 70)
print("PASO 5: Procesado individual por parking")
print("=" * 70)

audit_rows = []
parking_ids = sorted(df_raw["parking_id"].unique())

for pid in parking_ids:
    print(f"\n{'─'*50}")
    print(f"  Parking: {pid}")

    # --- 5.1 Filtrar y ordenar ---
    df_p = df_raw[df_raw["parking_id"] == pid].copy()
    nombre = df_p["parking_name"].iloc[0]
    capacidad = df_p["plazas_totales_oficial"].iloc[0]

    df_p = df_p[["timestamp", "ocupadas_oficial", "libres", "plazas_totales_oficial"]].copy()
    df_p = df_p.sort_values("timestamp")

    # --- 5.2 Bucketing a 15 min: quedarse con el último valor de cada slot ---
    df_p["ts_15"] = df_p["timestamp"].dt.floor(FREQ)
    df_p = df_p.groupby("ts_15").last().reset_index()
    df_p = df_p.drop(columns=["timestamp"]).rename(columns={"ts_15": "timestamp"})

    # --- 5.3 Reindexar a índice completo de 15min para exponer gaps ---
    ts_min = df_p["timestamp"].min()
    ts_max = df_p["timestamp"].max()
    idx_completo = pd.date_range(start=ts_min, end=ts_max, freq=FREQ)
    df_p = df_p.set_index("timestamp").reindex(idx_completo).reset_index()
    df_p.rename(columns={"index": "timestamp"}, inplace=True)

    slots_totales  = len(df_p)
    slots_con_dato = df_p["ocupadas_oficial"].notna().sum()
    cobertura_pct  = round(slots_con_dato / slots_totales * 100, 1)
    gaps_grandes   = (df_p["ocupadas_oficial"].isna()).sum()

    print(f"    Slots totales 15min: {slots_totales:,} | Con dato: {slots_con_dato:,} | Cobertura: {cobertura_pct}%")

    # --- 5.4 Merge meteorología (nearest backward — sin leakage futuro) ---
    df_meteo_sorted = df_meteo.sort_values("timestamp_local")
    df_p = pd.merge_asof(
        df_p.sort_values("timestamp"),
        df_meteo_sorted.rename(columns={"timestamp_local": "timestamp"}),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("1h"),  # si no hay dato meteo en 1h, queda NaN
    )

    # --- 5.5 Merge festivos ---
    df_p["fecha"] = df_p["timestamp"].dt.date
    df_p = df_p.merge(df_fest, on="fecha", how="left")
    df_p["festivo"] = df_p["festivo"].fillna(0).astype(int)
    df_p.drop(columns=["fecha"], inplace=True)

    # --- 5.6 Features de calendario ---
    df_p["hour"]       = df_p["timestamp"].dt.hour
    df_p["dow"]        = df_p["timestamp"].dt.dayofweek   # 0=lunes, 6=domingo
    df_p["month"]      = df_p["timestamp"].dt.month
    df_p["is_weekend"] = (df_p["dow"] >= 5).astype(int)

    # Temporada alta: junio-septiembre (flag binario)
    # Captura el patrón estacional de verano que 'month' continuo subestima
    df_p["mes_verano"] = df_p["month"].isin([6, 7, 8, 9]).astype(int)

    # Codificación cíclica (evita discontinuidad 23h→0h, dic→ene, etc.)
    df_p["hour_sin"]  = np.sin(2 * np.pi * df_p["hour"]  / 24)
    df_p["hour_cos"]  = np.cos(2 * np.pi * df_p["hour"]  / 24)
    df_p["dow_sin"]   = np.sin(2 * np.pi * df_p["dow"]   / 7)
    df_p["dow_cos"]   = np.cos(2 * np.pi * df_p["dow"]   / 7)
    df_p["month_sin"] = np.sin(2 * np.pi * df_p["month"] / 12)
    df_p["month_cos"] = np.cos(2 * np.pi * df_p["month"] / 12)

    # --- 5.6b Imputación de gaps breves del sensor (ffill, máx 2 slots = 30 min) ---
    # El sensor reporta con frecuencia real ~30 min (mediana entre lecturas válidas).
    # Slots vacíos entre lecturas se rellenan con el último valor conocido.
    # Límite 2 slots: cubre el 94% de los gaps sin inventar datos más allá de 30 min.
    # IMPORTANTE: 'ocupadas_oficial' original NO se modifica.
    #   → los targets (y_t15, y_t30…) siguen siendo honestos (solo slots con dato real)
    #   → las features (lags, rolling) sí usan el valor imputado para no anularse en gaps
    ocu_filled = df_p["ocupadas_oficial"].ffill(limit=2)

    # --- 5.7 Lags (sobre plazas ocupadas imputadas — sin leakage) ---
    # shift(1) garantiza que lag_1 es siempre el valor ANTERIOR, nunca el actual.
    # Se usa ocu_filled para que un gap de 1-2 slots no anule toda la cadena de lags.
    for lag in LAGS:
        df_p[f"ocu_lag_{lag}"] = ocu_filled.shift(lag)

    # Velocidad de cambio: diferencia entre los dos últimos valores observados
    # Captura si el parking está llenándose o vaciándose en tiempo real
    # Es clave para AN: permite anticipar rampas de subida bruscas
    df_p["ocu_delta_1"] = df_p["ocu_lag_1"] - df_p["ocu_lag_2"]

    # --- 5.8 Rolling (shift(1) antes del rolling para evitar leakage del propio valor) ---
    for window in ROLLING_WINDOWS:
        df_p[f"ocu_roll_mean_{window}"] = (
            ocu_filled.shift(1).rolling(window, min_periods=1).mean()
        )
        df_p[f"ocu_roll_std_{window}"] = (
            ocu_filled.shift(1).rolling(window, min_periods=2).std()
        )
        # Precipitación acumulada (también shifteada para evitar leakage)
        if "precipitation" in df_p.columns:
            df_p[f"precip_roll_sum_{window}"] = (
                df_p["precipitation"].shift(1).rolling(window, min_periods=1).sum()
            )

    # --- 5.9 Targets multi-horizonte ---
    # shift(-steps): valor FUTURO a steps pasos vista
    for horizon_name, steps in HORIZONS.items():
        df_p[f"y_{horizon_name}"] = df_p["ocupadas_oficial"].shift(-steps)

    # --- 5.10 Guardar capacidad como columna de referencia (no es feature) ---
    df_p["capacidad"] = capacidad

    # --- 5.11 Split temporal 70/15/15 ---
    # El split se determina con el horizonte más largo (el más restrictivo en NaN)
    # y el lag más largo. Usar todos los horizontes provocaría que añadir uno nuevo
    # (ej. t45) desplace los cortes train/valid/test al excluir filas extra en gaps.
    horizonte_largo = max(HORIZONS, key=HORIZONS.get)   # "t60" mientras sea el mayor
    df_valid_rows = df_p.dropna(subset=[f"y_{horizonte_largo}", f"ocu_lag_{LAGS[-1]}"])

    n_total = len(df_valid_rows)
    n_train = int(n_total * TRAIN_PCT)
    n_valid = int(n_total * VALID_PCT)
    n_test  = n_total - n_train - n_valid

    # Evaluación de suficiencia
    motivo_exclusion = None
    if n_total < MIN_ROWS_PARKING:
        motivo_exclusion = f"insuficientes registros válidos ({n_total} < {MIN_ROWS_PARKING})"
    elif n_test < MIN_TEST_ROWS:
        motivo_exclusion = f"test set demasiado pequeño ({n_test} < {MIN_TEST_ROWS} filas = {MIN_TEST_ROWS//96//7} semanas)"

    decision = "excluir_datos" if motivo_exclusion else "incluir"

    # Índices de corte sobre el dataframe ordenado
    idx_train_end = df_valid_rows.index[n_train - 1]
    idx_valid_end = df_valid_rows.index[n_train + n_valid - 1]

    df_p["split"] = None
    df_p.loc[df_p.index <= idx_train_end, "split"] = "train"
    df_p.loc[(df_p.index > idx_train_end) & (df_p.index <= idx_valid_end), "split"] = "valid"
    df_p.loc[df_p.index > idx_valid_end, "split"] = "test"

    print(f"    Registros válidos: {n_total:,} → train={n_train:,} | valid={n_valid:,} | test={n_test:,}")
    print(f"    Decisión: {decision}" + (f" — {motivo_exclusion}" if motivo_exclusion else ""))

    # --- 5.12 Guardar por parking ---
    out_path = PROC_DIR / f"{pid}.pkl"
    df_p.to_pickle(out_path)
    print(f"    Guardado → {out_path.name}")

    audit_rows.append({
        "parking_id"       : pid,
        "parking_name"     : nombre,
        "capacidad"        : capacidad,
        "slots_totales"    : slots_totales,
        "slots_con_dato"   : slots_con_dato,
        "cobertura_pct"    : cobertura_pct,
        "gaps_slots"       : gaps_grandes,
        "registros_validos": n_total,
        "n_train"          : n_train,
        "n_valid"          : n_valid,
        "n_test"           : n_test,
        "decision"         : decision,
        "motivo_exclusion" : motivo_exclusion or "",
    })


# =============================================================================
# 6. REPORTE DE AUDITORÍA
# =============================================================================
print("\n" + "=" * 70)
print("PASO 6: Reporte de auditoría")
print("=" * 70)

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUT_DIR / "audit_parkings.csv", index=False)

print("\n📋 AUDITORÍA DE PARKINGS:")
print(audit_df[[
    "parking_id","parking_name","capacidad","cobertura_pct",
    "registros_validos","n_train","n_valid","n_test","decision"
]].to_string(index=False))

n_incluidos = (audit_df["decision"] == "incluir").sum()
n_excluidos = (audit_df["decision"] != "incluir").sum()
print(f"\n✅ Parkings incluidos: {n_incluidos}")
print(f"❌ Parkings excluidos: {n_excluidos}")
print("\n✅ Preprocesado completado")
