# =============================================================================
# 03_validate.py — Validación completa y outputs por parking
#
# Input:  outputs/processed/<pid>.parquet
#         outputs/models/<pid>_<horizonte>.joblib
#         outputs/metrics_entrenamiento.csv
#
# Output:
#   outputs/validation/resumen_global.csv       → tabla maestra cross-parking
#   outputs/validation/<pid>_<horizonte>.csv    → detalle por parking/horizonte
#   outputs/validation/error_por_hora.csv       → MAE por hora (todos los parkings)
#   outputs/validation/shap/<pid>_<h>_shap.csv → importancias SHAP
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # sin display (servidor)
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    OUT_DIR, HORIZONS, PEAK_THRESHOLD_PCT,
    TRAMOS_OCUPACION, TOP_N_FEATURES, SHAP_SAMPLE_N,
    SATURACION_UMBRAL_ROJO, SATURACION_UMBRAL_AMARILLO,
)

PROC_DIR   = OUT_DIR / "processed"
MODELS_DIR = OUT_DIR / "models"
VAL_DIR    = OUT_DIR / "validation"
SHAP_DIR   = VAL_DIR / "shap"
PLOTS_DIR  = VAL_DIR / "plots"

for d in [VAL_DIR, SHAP_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def compute_recall_peak(y_true: np.ndarray, y_pred: np.ndarray, umbral: float) -> float:
    """Recall: de los picos reales, qué % detecta el modelo."""
    picos_reales = y_true >= umbral
    if picos_reales.sum() == 0:
        return np.nan
    return (picos_reales & (y_pred >= umbral)).sum() / picos_reales.sum()


def compute_precision_peak(y_true: np.ndarray, y_pred: np.ndarray, umbral: float) -> float:
    """Precisión: de las alertas del modelo, qué % son picos reales."""
    picos_pred = y_pred >= umbral
    if picos_pred.sum() == 0:
        return np.nan
    return (picos_pred & (y_true >= umbral)).sum() / picos_pred.sum()


def error_direccional_en_pico(y_true: np.ndarray, y_pred: np.ndarray, umbral: float) -> dict:
    """En picos reales: error medio y dirección (+ = supra-pred, - = infra-pred)."""
    mask = y_true >= umbral
    if mask.sum() == 0:
        return {"n_picos": 0, "error_medio_pico": np.nan, "infrapred_pct": np.nan}
    err = y_pred[mask] - y_true[mask]
    return {
        "n_picos"           : int(mask.sum()),
        "error_medio_pico"  : round(float(err.mean()), 2),
        "infrapred_pct"     : round(float((err < 0).mean() * 100), 1),  # % veces infra-pred
    }


def tramo_ocupacion(val: float, capacidad: float) -> str:
    """Asigna un tramo de ocupación según % de capacidad."""
    pct = val / capacidad
    for nombre, (lo, hi) in TRAMOS_OCUPACION.items():
        if lo <= pct < hi:
            return nombre
    return "alto"


def plot_real_vs_pred(ts, y_true, y_pred, pid, horizonte, out_path: Path):
    """Gráfica de serie temporal real vs predicción (primeras 2 semanas del test)."""
    n = min(len(ts), 2 * 7 * 96)   # 2 semanas máx
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts[:n], y_true[:n], label="Real", linewidth=1.2)
    ax.plot(ts[:n], y_pred[:n], label="Pred", linewidth=1.0, linestyle="--", alpha=0.85)
    ax.set_title(f"{pid} — {horizonte.upper()} | Real vs Predicción (2 semanas test)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Plazas ocupadas")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def compute_semaforo(pred: np.ndarray, capacidad: float) -> tuple:
    """
    Calcula el semáforo de saturación para cualquier horizonte de predicción.

    Lógica:
      delta[t]          = pred[t] - pred[t-1]          (coches/slot, 1 slot = 15 min)
      espacios_libres   = capacidad - pred[t]
      slots_hasta_sat   = espacios_libres / delta       (si delta > 0, si no → inf)

    Umbrales (configurables en config.py):
      rojo     → slots_hasta_sat < SATURACION_UMBRAL_ROJO     (< 1 hora)
      amarillo → slots_hasta_sat < SATURACION_UMBRAL_AMARILLO  (< 2 horas)
      verde    → resto (vaciándose o más de 2 horas para saturarse)

    Aplicable a t15, t30, t45 y t60: el delta mide la velocidad de cambio
    de la predicción de ese horizonte entre slots consecutivos.

    Returns:
      delta          : np.ndarray  — velocidad de cambio por slot
      slots_hasta_sat: np.ndarray  — slots hasta saturación (np.inf si no hay riesgo)
      semaforo       : np.ndarray  — array de strings: "verde"/"amarillo"/"rojo"
    """
    # Delta: diferencia con el slot anterior; el primer elemento = 0 (sin referencia)
    delta = np.diff(pred, prepend=pred[0])
    delta[0] = 0.0

    espacios_libres = np.maximum(capacidad - pred, 0.0)

    # slots_hasta_sat solo tiene sentido cuando el parking se está llenando (delta > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slots_hasta_sat = np.where(delta > 0, espacios_libres / delta, np.inf)

    # Clasificación
    semaforo = np.where(
        slots_hasta_sat < SATURACION_UMBRAL_ROJO, "rojo",
        np.where(slots_hasta_sat < SATURACION_UMBRAL_AMARILLO, "amarillo", "verde")
    )

    return delta, slots_hasta_sat, semaforo


def validate_parking_horizon(pid: str, horizonte: str, df: pd.DataFrame) -> dict | None:
    """Valida un modelo parking+horizonte. Devuelve dict de métricas."""
    model_path = MODELS_DIR / f"{pid}_{horizonte}.joblib"
    feat_path  = MODELS_DIR / f"{pid}_feature_cols.json"

    if not model_path.exists():
        return None

    model = joblib.load(model_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    ycol     = f"y_{horizonte}"
    test_df  = df[df["split"] == "test"].dropna(subset=feature_cols + [ycol])

    if len(test_df) < 20:
        print(f"    ⚠️  Test insuficiente para {pid}/{horizonte}")
        return None

    capacidad = test_df["capacidad"].iloc[0]
    umbral    = capacidad * PEAK_THRESHOLD_PCT

    X_test = test_df[feature_cols]
    y_test = test_df[ycol].values
    ts_test = test_df["timestamp"].values if "timestamp" in test_df.columns else None

    # Predicción con clipping físico
    pred_raw = model.predict(X_test)
    pred     = np.clip(pred_raw, 0, capacidad)

    # ── Métricas principales ──────────────────────────────────────────────────
    mae   = mean_absolute_error(y_test, pred)
    rmse  = np.sqrt(mean_squared_error(y_test, pred))
    r2    = r2_score(y_test, pred)
    mape_cap = mae / capacidad * 100

    # Baseline lag_1
    if "ocu_lag_1" in test_df.columns:
        bl = np.clip(test_df["ocu_lag_1"].values, 0, capacidad)
        mask_bl = ~np.isnan(bl)
        baseline_mae  = mean_absolute_error(y_test[mask_bl], bl[mask_bl]) if mask_bl.sum() else np.nan
        baseline_rmse = np.sqrt(mean_squared_error(y_test[mask_bl], bl[mask_bl])) if mask_bl.sum() else np.nan
        mejora_mae = (baseline_mae - mae) / baseline_mae * 100 if (not np.isnan(baseline_mae) and baseline_mae > 0) else np.nan
    else:
        baseline_mae = baseline_rmse = mejora_mae = np.nan

    # Recall y precisión en picos
    recall_pico    = compute_recall_peak(y_test, pred, umbral)
    precision_pico = compute_precision_peak(y_test, pred, umbral)
    dir_pico       = error_direccional_en_pico(y_test, pred, umbral)

    # ── Error por tramo de ocupación ─────────────────────────────────────────
    tramos_result = {}
    for nombre_tramo, (lo, hi) in TRAMOS_OCUPACION.items():
        mask_t = (y_test / capacidad >= lo) & (y_test / capacidad < hi)
        if mask_t.sum() > 0:
            tramos_result[f"mae_{nombre_tramo}"] = round(float(mean_absolute_error(y_test[mask_t], pred[mask_t])), 2)
            tramos_result[f"n_{nombre_tramo}"]   = int(mask_t.sum())

    # ── Error por hora ───────────────────────────────────────────────────────
    if "hour" in test_df.columns:
        err_hora = pd.DataFrame({
            "hour"      : test_df["hour"].values,
            "abs_error" : np.abs(y_test - pred),
        }).groupby("hour").agg(mae=("abs_error", "mean"), n=("abs_error", "count"))
        err_hora["parking_id"] = pid
        err_hora["horizonte"]  = horizonte
        err_hora = err_hora.reset_index()
    else:
        err_hora = None

    # ── SHAP ─────────────────────────────────────────────────────────────────
    shap_importance = None
    try:
        sample_n  = min(SHAP_SAMPLE_N, len(X_test))
        X_shap    = X_test.sample(sample_n, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_shap)
        shap_abs  = np.abs(shap_vals).mean(axis=0)
        shap_importance = pd.Series(shap_abs, index=X_shap.columns).sort_values(ascending=False)
    except Exception as e:
        print(f"    ⚠️  SHAP falló para {pid}/{horizonte}: {e}")

    # ── Feature importance nativa ────────────────────────────────────────────
    fi_series = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # ── Gráfica real vs predicción ───────────────────────────────────────────
    if ts_test is not None:
        plot_path = PLOTS_DIR / f"{pid}_{horizonte}_real_vs_pred.png"
        plot_real_vs_pred(ts_test, y_test, pred, pid, horizonte, plot_path)

    # ── Detalle fila a fila — test COMPLETO (para el mapa) ──────────────────
    # Se usa un conjunto más amplio que el estricto de métricas: solo requiere
    # que la ocupación (target) sea válida. Donde faltan features, y_pred = NaN
    # y el semáforo se marca "nodata". Así el mapa tiene un timestamp por cada
    # slot con dato real, no solo los slots con features meteorológicas completas.
    test_all = df[df["split"] == "test"].dropna(subset=[ycol]).copy()
    y_all    = test_all[ycol].values
    ts_all   = test_all["timestamp"].values if "timestamp" in test_all.columns else None

    # Predecir solo donde todos los features están disponibles
    can_pred = test_all[feature_cols].notna().all(axis=1)
    pred_all = np.full(len(test_all), np.nan)
    if can_pred.sum() > 0:
        pred_all[can_pred.values] = np.clip(
            model.predict(test_all.loc[can_pred, feature_cols]), 0, capacidad
        )

    detail_df = pd.DataFrame({
        "timestamp"    : ts_all,
        "y_real"       : y_all,
        "y_pred"       : pred_all,
        "abs_error"    : np.where(np.isnan(pred_all), np.nan, np.abs(y_all - pred_all)),
        "error_sign"   : np.where(np.isnan(pred_all), np.nan, pred_all - y_all),
        "tramo_real"   : [tramo_ocupacion(v, capacidad) for v in y_all],
        "en_pico_real" : y_all >= umbral,
        "en_pico_pred" : np.where(np.isnan(pred_all), False, pred_all >= umbral),
    })

    # Semáforo: se calcula sobre pred_all con ffill para cubrir huecos entre
    # slots sin features. Los slots sin predicción se marcan "nodata".
    pred_for_sem = pd.Series(pred_all).ffill().bfill().fillna(0.0).values
    delta_all, slots_sat_all, semaforo_all = compute_semaforo(pred_for_sem, capacidad)

    detail_df["delta_pred"]      = np.where(np.isnan(pred_all), np.nan, delta_all)
    detail_df["slots_hasta_sat"] = np.where(
        np.isnan(pred_all) | np.isinf(slots_sat_all), -1.0,
        np.round(slots_sat_all, 2)
    )
    detail_df["semaforo"] = np.where(np.isnan(pred_all), "nodata", semaforo_all)

    detail_df.to_csv(VAL_DIR / f"{pid}_{horizonte}_detalle.csv", index=False)

    # Exportar SHAP
    if shap_importance is not None:
        shap_df = shap_importance.reset_index()
        shap_df.columns = ["feature", "mean_abs_shap"]
        shap_df.to_csv(SHAP_DIR / f"{pid}_{horizonte}_shap.csv", index=False)

    print(f"    {horizonte.upper()}: MAE={mae:.1f}pz | MAPE={mape_cap:.1f}% | "
          f"Recall@{int(PEAK_THRESHOLD_PCT*100)}%={recall_pico:.3f} | "
          f"Mejora_MAE={mejora_mae:.1f}% vs baseline | R²={r2:.3f}")

    return {
        "parking_id"        : pid,
        "horizonte"         : horizonte,
        "capacidad"         : int(capacidad),
        "n_test"            : len(test_df),
        "mae_plazas"        : round(mae, 2),
        "rmse_plazas"       : round(rmse, 2),
        "r2"                : round(r2, 4),
        "mape_cap_pct"      : round(mape_cap, 2),
        "baseline_mae"      : round(baseline_mae, 2) if not np.isnan(baseline_mae) else None,
        "mejora_mae_pct"    : round(mejora_mae, 1) if not np.isnan(mejora_mae) else None,
        f"recall_{int(PEAK_THRESHOLD_PCT*100)}pct": round(recall_pico, 4) if not np.isnan(recall_pico) else None,
        f"precision_{int(PEAK_THRESHOLD_PCT*100)}pct": round(precision_pico, 4) if not np.isnan(precision_pico) else None,
        "n_picos_test"      : dir_pico["n_picos"],
        "error_medio_pico"  : dir_pico["error_medio_pico"],
        "infrapred_pico_pct": dir_pico["infrapred_pct"],
        **tramos_result,
        "_err_hora_df"      : err_hora,       # interno, se extrae después
        "_shap_top5"        : list(shap_importance.head(5).index) if shap_importance is not None else [],
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VALIDACIÓN COMPLETA — PER PARKING")
    print("=" * 70)

    proc_files = sorted(PROC_DIR.glob("*.pkl"))
    if not proc_files:
        raise FileNotFoundError(f"No hay .pkl en {PROC_DIR}. Ejecuta primero los scripts anteriores.")

    all_results  = []
    all_err_hora = []

    for pq_file in proc_files:
        pid = pq_file.stem
        df  = pd.read_pickle(pq_file)

        if "split" not in df.columns or df["split"].isna().all():
            print(f"\n⏭️  {pid}: sin split → saltando")
            continue

        print(f"\n{'─'*60}")
        print(f"  PARKING: {pid}")

        for horizonte in HORIZONS:
            result = validate_parking_horizon(pid, horizonte, df)
            if result is None:
                continue

            # Extraer componentes internos antes de guardar la fila
            err_hora_df = result.pop("_err_hora_df")
            result.pop("_shap_top5")   # ya está en CSV de SHAP

            if err_hora_df is not None:
                all_err_hora.append(err_hora_df)

            all_results.append(result)

    # ── Resumen global ────────────────────────────────────────────────────────
    resumen_df = pd.DataFrame(all_results)
    resumen_df = resumen_df.sort_values(["parking_id", "horizonte"])
    resumen_df.to_csv(VAL_DIR / "resumen_global.csv", index=False)

    # ── Error por hora consolidado ────────────────────────────────────────────
    if all_err_hora:
        err_hora_df = pd.concat(all_err_hora, ignore_index=True)
        err_hora_df.to_csv(VAL_DIR / "error_por_hora.csv", index=False)

    # ── Imprimir tabla maestra ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLA MAESTRA DE RESULTADOS")
    print("=" * 70)

    display_cols = [
        "parking_id", "horizonte", "capacidad",
        "mae_plazas", "mape_cap_pct",
        "recall_85pct", "precision_85pct",
        "mejora_mae_pct", "r2",
    ]
    display_cols = [c for c in display_cols if c in resumen_df.columns]
    print(resumen_df[display_cols].to_string(index=False))

    # ── Ranking por recall en picos (métrica principal) ───────────────────────
    recall_col = f"recall_{int(PEAK_THRESHOLD_PCT*100)}pct"
    if recall_col in resumen_df.columns:
        print(f"\n📊 RANKING POR RECALL@{int(PEAK_THRESHOLD_PCT*100)}% (métrica principal):")
        ranking = (
            resumen_df.groupby("parking_id")[recall_col]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        ranking.columns = ["parking_id", f"recall_{int(PEAK_THRESHOLD_PCT*100)}pct_medio"]
        print(ranking.to_string(index=False))

    print(f"\n✅ Validación completada")
    print(f"   Resultados en: {VAL_DIR}")
