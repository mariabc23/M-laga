# =============================================================================
# 02_train.py — Entrenamiento per-parking con XGBoost quantile loss
#
# Input:  outputs/processed/<parking_id>.parquet  (generado por 01_preprocessing.py)
# Output: outputs/models/<parking_id>_<horizonte>.joblib
#         outputs/models/<parking_id>_feature_cols.json
#         outputs/metrics_entrenamiento.csv
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    OUT_DIR, PARKING_ID, TARGET_COL, CAPACITY_COL,
    HORIZONS, XGB_PARAMS, ALPHA_GLOBAL, ALPHA_POR_PARKING,
    PEAK_THRESHOLD_PCT, TOP_N_FEATURES,
)

PROC_DIR   = OUT_DIR / "processed"
MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Columnas que NUNCA son features (metadatos, targets, referencias)
EXCLUDE_FROM_FEATURES = [
    "timestamp", "ocupadas_oficial", "libres",
    "plazas_totales_oficial", "capacidad", "split",
] + [f"y_{h}" for h in HORIZONS]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas válidas como features del modelo."""
    candidates = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    # Verificar que no queden columnas tipo object
    obj_cols = df[candidates].select_dtypes("object").columns.tolist()
    if obj_cols:
        print(f"  ⚠️  Columnas object eliminadas de features: {obj_cols}")
        candidates = [c for c in candidates if c not in obj_cols]
    return candidates


def train_parking(pid: str, df: pd.DataFrame) -> list[dict]:
    """Entrena un modelo por horizonte para un parking. Devuelve lista de métricas."""
    print(f"\n{'═'*60}")
    print(f"  PARKING: {pid}  ({df['split'].value_counts().to_dict()})")
    print(f"{'═'*60}")

    capacidad = df["capacidad"].iloc[0]
    alpha = ALPHA_POR_PARKING.get(pid, ALPHA_GLOBAL)
    if pid in ALPHA_POR_PARKING:
        print(f"  ℹ️  Alpha override para {pid}: {alpha}")

    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df  = df[df["split"] == "test"]

    feature_cols = get_feature_cols(df)

    # Guardar lista de features de este parking
    with open(MODELS_DIR / f"{pid}_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    metrics_parking = []

    for horizon_name, steps in HORIZONS.items():
        ycol = f"y_{horizon_name}"
        print(f"\n  ── Horizonte {horizon_name.upper()} ──")

        # Eliminar NaN en features y en el target de este horizonte
        needed = feature_cols + [ycol]
        tr = train_df.dropna(subset=needed)
        va = valid_df.dropna(subset=needed)
        te = test_df.dropna(subset=needed)

        if len(tr) < 100 or len(va) < 20 or len(te) < 20:
            print(f"  ⚠️  Datos insuficientes para {pid}/{horizon_name} → saltando")
            continue

        X_train, y_train = tr[feature_cols], tr[ycol]
        X_valid, y_valid = va[feature_cols], va[ycol]
        X_test,  y_test  = te[feature_cols], te[ycol]

        # Construir parámetros con alpha específico
        params = {**XGB_PARAMS, "quantile_alpha": alpha}

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        # Predicción y clipping a rango físico válido [0, capacidad]
        pred_raw = model.predict(X_test)
        pred     = np.clip(pred_raw, 0, capacidad)

        # Métricas básicas
        mae  = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)

        # MAPE normalizado por capacidad (para comparar entre parkings)
        mape_cap = mae / capacidad * 100

        # Baseline: predecir lag_1 (ocupación hace 15 min)
        lag1_col = "ocu_lag_1"
        if lag1_col in te.columns:
            baseline = np.clip(te[lag1_col].values, 0, capacidad)
            # Alinear índices con y_test
            base_series = te[lag1_col].reindex(y_test.index)
            baseline_aligned = np.clip(base_series.values, 0, capacidad)
            mask_base = ~np.isnan(baseline_aligned)
            if mask_base.sum() > 0:
                baseline_mae  = mean_absolute_error(y_test.values[mask_base], baseline_aligned[mask_base])
                baseline_rmse = np.sqrt(mean_squared_error(y_test.values[mask_base], baseline_aligned[mask_base]))
            else:
                baseline_mae = baseline_rmse = np.nan
        else:
            baseline_mae = baseline_rmse = np.nan

        # Recall en picos (85% de capacidad)
        umbral_plazas  = capacidad * PEAK_THRESHOLD_PCT
        es_pico_real   = y_test.values >= umbral_plazas
        es_pico_pred   = pred >= umbral_plazas
        n_picos_reales = es_pico_real.sum()
        if n_picos_reales > 0:
            recall_pico = (es_pico_real & es_pico_pred).sum() / n_picos_reales
        else:
            recall_pico = np.nan

        # Guardado del modelo
        model_path = MODELS_DIR / f"{pid}_{horizon_name}.joblib"
        joblib.dump(model, model_path)

        print(f"    MAE={mae:.1f} plazas | MAPE_cap={mape_cap:.1f}% | RMSE={rmse:.1f} | R²={r2:.3f}")
        print(f"    Recall@{int(PEAK_THRESHOLD_PCT*100)}%={recall_pico:.3f} | Baseline MAE={baseline_mae:.1f} | best_iter={model.best_iteration}")

        metrics_parking.append({
            "parking_id"    : pid,
            "horizonte"     : horizon_name,
            "capacidad"     : capacidad,
            "alpha"         : alpha,
            "n_train"       : len(tr),
            "n_valid"       : len(va),
            "n_test"        : len(te),
            "mae_plazas"    : round(mae, 2),
            "rmse_plazas"   : round(rmse, 2),
            "r2"            : round(r2, 4),
            "mape_cap_pct"  : round(mape_cap, 2),
            "baseline_mae"  : round(baseline_mae, 2) if not np.isnan(baseline_mae) else None,
            "baseline_rmse" : round(baseline_rmse, 2) if not np.isnan(baseline_rmse) else None,
            "mejora_mae_pct": round((baseline_mae - mae) / baseline_mae * 100, 1) if (not np.isnan(baseline_mae) and baseline_mae > 0) else None,
            f"recall_{int(PEAK_THRESHOLD_PCT*100)}pct": round(recall_pico, 4) if not np.isnan(recall_pico) else None,
            "best_iteration": model.best_iteration,
        })

    return metrics_parking


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ENTRENAMIENTO PER-PARKING — XGBoost Quantile Loss")
    print("=" * 70)

    proc_files = sorted(PROC_DIR.glob("*.pkl"))
    if not proc_files:
        raise FileNotFoundError(f"No hay archivos .pkl en {PROC_DIR}. Ejecuta primero 01_preprocessing.py")

    all_metrics = []
    skipped = []

    for pq_file in proc_files:
        pid = pq_file.stem
        df  = pd.read_pickle(pq_file)

        # Solo entrenar parkings marcados como 'incluir' en el split
        if df["split"].isna().all():
            print(f"\n⏭️  {pid}: sin split definido → saltando")
            skipped.append(pid)
            continue

        metrics = train_parking(pid, df)
        all_metrics.extend(metrics)

    # ─── Tabla de métricas consolidada ───
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(["parking_id", "horizonte"])
    metrics_df.to_csv(OUT_DIR / "metrics_entrenamiento.csv", index=False)

    print("\n" + "=" * 70)
    print("RESUMEN DE ENTRENAMIENTO")
    print("=" * 70)
    print(metrics_df[[
        "parking_id","horizonte","mae_plazas","mape_cap_pct",
        "recall_85pct","mejora_mae_pct","r2","best_iteration"
    ]].to_string(index=False))

    if skipped:
        print(f"\n⏭️  Parkings saltados: {skipped}")

    print("\n✅ Entrenamiento completado")
    print(f"   Modelos guardados en: {MODELS_DIR}")
    print(f"   Métricas guardadas en: {OUT_DIR / 'metrics_entrenamiento.csv'}")
