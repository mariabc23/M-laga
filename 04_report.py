# =============================================================================
# 04_report.py — Dashboard HTML consolidado por parking
#
# Genera un único archivo HTML interactivo que combina:
#   - Tabla resumen global (todos los parkings, todas las métricas)
#   - Por cada parking: gráfica real vs pred + tabla de métricas + top SHAP
#
# Abrir en cualquier navegador: outputs/reporte_parkings.html
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import base64
import io
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from config import (
    OUT_DIR, HORIZONS, PEAK_THRESHOLD_PCT,
    ALPHA_GLOBAL, ALPHA_POR_PARKING, XGB_PARAMS,
    SATURACION_UMBRAL_ROJO, SATURACION_UMBRAL_AMARILLO,
)

VAL_DIR   = OUT_DIR / "validation"
MODELS_DIR = OUT_DIR / "models"
PROC_DIR  = OUT_DIR / "processed"

HORIZON_LABELS = {"t15": "15 min", "t30": "30 min", "t45": "45 min", "t60": "60 min"}
PARKING_NAMES  = {}   # se rellena dinámicamente desde parking_reference.csv


# =============================================================================
# HELPERS
# =============================================================================

def fig_to_b64(fig) -> str:
    """Convierte figura matplotlib a string base64 para incrustar en HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def color_recall(val) -> str:
    """Color semáforo según recall@85%."""
    if pd.isna(val):
        return "#888888"
    if val >= 0.75:
        return "#27ae60"   # verde
    if val >= 0.50:
        return "#f39c12"   # naranja
    return "#e74c3c"       # rojo


def color_mejora(val) -> str:
    if pd.isna(val):
        return "#888888"
    return "#27ae60" if val > 0 else "#e74c3c"


def make_horizon_figure(pid: str, h: str, df: pd.DataFrame,
                         shap_df: pd.DataFrame, capacidad: float) -> str:
    """
    Genera una figura para UN horizonte: gráfica real vs pred (ancho completo)
    más top SHAP debajo. Devuelve imagen base64.
    """
    umbral = capacidad * PEAK_THRESHOLD_PCT
    df = df.copy()

    fig = plt.figure(figsize=(20, 8))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[3, 1])

    # ── Gráfica real vs pred ──────────────────────────────────────────────────
    ax_plot = fig.add_subplot(gs[0])
    n_pts   = min(len(df), 7 * 96)   # 1 semana = 672 slots de 15 min

    if "timestamp" in df.columns:
        x_vals  = pd.to_datetime(df["timestamp"].values[:n_pts])
        x_label = "Fecha"
    else:
        x_vals  = np.arange(n_pts)
        x_label = "Slots (1 sem. test)"

    y_real = df["y_real"].values[:n_pts]
    y_pred = df["y_pred"].values[:n_pts]

    ax_plot.plot(x_vals, y_real, label="Real",  lw=1.4, color="#2c3e50")
    ax_plot.plot(x_vals, y_pred, label="Pred",  lw=1.1,
                 color="#e74c3c", linestyle="--", alpha=0.85)
    ax_plot.axhline(umbral, color="#f39c12", lw=1.0, linestyle=":",
                    label=f"Umbral {int(PEAK_THRESHOLD_PCT*100)}%")

    # Sombreado en zonas de pico real
    in_peak = False
    peak_start = None
    for i in range(n_pts):
        val = y_real[i]
        if not np.isnan(val) and val >= umbral and not in_peak:
            in_peak     = True
            peak_start  = x_vals[i]
        elif (np.isnan(val) or val < umbral) and in_peak:
            ax_plot.axvspan(peak_start, x_vals[i],
                            alpha=0.13, color="#e74c3c", linewidth=0)
            in_peak = False
    if in_peak:
        ax_plot.axvspan(peak_start, x_vals[-1],
                        alpha=0.13, color="#e74c3c", linewidth=0)

    ax_plot.set_title(
        f"{pid}  —  {PARKING_NAMES.get(pid, '')}  |  "
        f"Horizonte: {HORIZON_LABELS.get(h, h)}  |  Capacidad: {int(capacidad)} plazas",
        fontsize=11, fontweight="bold"
    )
    ax_plot.set_ylabel("Plazas ocupadas", fontsize=9)
    ax_plot.set_xlabel(x_label, fontsize=8)
    ax_plot.legend(fontsize=8, loc="upper left")
    ax_plot.tick_params(labelsize=8)
    if not isinstance(x_vals, np.ndarray) or x_vals.dtype.kind == "M":
        fig.autofmt_xdate(rotation=30, ha="right")
    else:
        ax_plot.set_xlim(0, n_pts)

    # ── SHAP top features ─────────────────────────────────────────────────────
    ax_shap = fig.add_subplot(gs[1])
    if shap_df is not None and not shap_df.empty:
        top = shap_df.head(10)
        ax_shap.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
                     color="#3498db", edgecolor="white", height=0.65)
        ax_shap.set_title("Top 10 SHAP features", fontsize=9)
        ax_shap.set_xlabel("|SHAP| medio", fontsize=8)
        ax_shap.tick_params(labelsize=7)
    else:
        ax_shap.text(0.5, 0.5, "SHAP no disponible",
                     ha="center", va="center", transform=ax_shap.transAxes, fontsize=9)
        ax_shap.axis("off")

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_tabs_html(pid: str, images: dict) -> str:
    """
    Genera el selector de horizontes con tabs HTML/JS.
    images = {"t15": b64_str, "t30": b64_str, "t60": b64_str}
    """
    if not images:
        return ""

    # Botones de tab
    btn_html = ""
    for i, h in enumerate(HORIZONS):
        if h not in images:
            continue
        active = "tab-btn active" if i == 0 else "tab-btn"
        label  = HORIZON_LABELS.get(h, h)
        btn_html += f'<button class="{active}" onclick="showTab(\'{pid}\',\'{h}\',this)">{label}</button>\n'

    # Paneles con imágenes
    panels_html = ""
    for i, h in enumerate(HORIZONS):
        if h not in images:
            continue
        display = "block" if i == 0 else "none"
        panels_html += (
            f'<div id="tab-{pid}-{h}" class="tab-panel" style="display:{display}">'
            f'<img src="data:image/png;base64,{images[h]}" style="width:100%"/>'
            f'</div>\n'
        )

    return f"""
    <div class="tab-container">
      <div class="tab-bar">{btn_html}</div>
      {panels_html}
    </div>"""


def make_metrics_table_html(pid: str, resumen: pd.DataFrame) -> str:
    """Tabla HTML de métricas por horizonte para un parking."""
    park_df = resumen[resumen["parking_id"] == pid].copy()
    if park_df.empty:
        return "<p><em>Sin métricas disponibles.</em></p>"

    recall_col = f"recall_{int(PEAK_THRESHOLD_PCT*100)}pct"
    prec_col   = f"precision_{int(PEAK_THRESHOLD_PCT*100)}pct"
    mejora_col = "mejora_mae_pct"

    rows_html = ""
    for _, row in park_df.iterrows():
        h        = row.get("horizonte", "")
        mae      = f"{row.get('mae_plazas', float('nan')):.1f}"
        mape     = f"{row.get('mape_cap_pct', float('nan')):.1f}%"
        recall   = row.get(recall_col, float("nan"))
        prec     = row.get(prec_col, float("nan"))
        mejora   = row.get(mejora_col, float("nan"))
        r2       = f"{row.get('r2', float('nan')):.3f}"
        base_mae = f"{row.get('baseline_mae', float('nan')):.1f}"
        n_picos  = int(row.get("n_picos_test", 0))

        recall_str = f"{recall:.1%}" if not pd.isna(recall) else "—"
        prec_str   = f"{prec:.1%}"   if not pd.isna(prec)   else "—"
        mejora_str = f"{mejora:+.1f}%" if not pd.isna(mejora) else "—"

        rc = color_recall(recall)
        mc = color_mejora(mejora if not pd.isna(mejora) else None)

        rows_html += f"""
        <tr>
          <td><strong>{HORIZON_LABELS.get(h, h)}</strong></td>
          <td>{mae} plazas</td>
          <td>{mape}</td>
          <td style="color:{rc}; font-weight:bold">{recall_str}</td>
          <td>{prec_str}</td>
          <td style="color:{mc}">{mejora_str}</td>
          <td>{base_mae} plazas</td>
          <td>{r2}</td>
          <td>{n_picos}</td>
        </tr>"""

    return f"""
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Horizonte</th>
          <th>MAE</th>
          <th>MAPE</th>
          <th>Recall@{int(PEAK_THRESHOLD_PCT*100)}%</th>
          <th>Precisión@{int(PEAK_THRESHOLD_PCT*100)}%</th>
          <th>Mejora vs Baseline</th>
          <th>Baseline MAE</th>
          <th>R²</th>
          <th>Picos en test</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def make_global_table_html(resumen: pd.DataFrame) -> str:
    """Tabla resumen global cross-parking (media por parking)."""
    recall_col = f"recall_{int(PEAK_THRESHOLD_PCT*100)}pct"
    mejora_col = "mejora_mae_pct"

    agg = resumen.groupby("parking_id").agg(
        nombre        = ("parking_id", "first"),
        mae_medio     = ("mae_plazas", "mean"),
        mape_medio    = ("mape_cap_pct", "mean"),
        recall_medio  = (recall_col, "mean") if recall_col in resumen.columns else ("mae_plazas", "mean"),
        mejora_media  = (mejora_col, "mean") if mejora_col in resumen.columns else ("mae_plazas", "mean"),
        r2_medio      = ("r2", "mean"),
    ).reset_index().sort_values("recall_medio", ascending=False)

    # Añadir nombre del parking
    agg["nombre"] = agg["parking_id"].map(lambda x: PARKING_NAMES.get(x, x))

    rows_html = ""
    for _, row in agg.iterrows():
        rc = color_recall(row.get("recall_medio"))
        mc = color_mejora(row.get("mejora_media"))
        rows_html += f"""
        <tr>
          <td><strong>{row['parking_id']}</strong></td>
          <td>{row['nombre']}</td>
          <td>{row['mae_medio']:.1f} plazas</td>
          <td>{row['mape_medio']:.1f}%</td>
          <td style="color:{rc}; font-weight:bold">{row['recall_medio']:.1%}</td>
          <td style="color:{mc}">{row['mejora_media']:+.1f}%</td>
          <td>{row['r2_medio']:.3f}</td>
        </tr>"""

    return f"""
    <table class="metrics-table">
      <thead>
        <tr>
          <th>ID</th><th>Parking</th><th>MAE medio</th><th>MAPE medio</th>
          <th>Recall@{int(PEAK_THRESHOLD_PCT*100)}% medio ↓</th>
          <th>Mejora vs Baseline</th><th>R² medio</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def make_hyperparams_table_html(parking_ids: list, metrics_df: pd.DataFrame) -> str:
    """
    Tabla de hiperparámetros activos por parking.
    Muestra alpha, umbral de pico, parámetros XGBoost clave y features especiales.
    """
    n_est   = XGB_PARAMS.get("n_estimators", "—")
    depth   = XGB_PARAMS.get("max_depth", "—")
    lr      = XGB_PARAMS.get("learning_rate", "—")
    subsamp = XGB_PARAMS.get("subsample", "—")

    rows_html = ""
    for pid in parking_ids:
        nombre = PARKING_NAMES.get(pid, pid)
        alpha  = ALPHA_POR_PARKING.get(pid, ALPHA_GLOBAL)
        umbral = f"{int(PEAK_THRESHOLD_PCT * 100)}%"

        # Override badge
        alpha_style = "font-weight:bold; color:#e67e22" if pid in ALPHA_POR_PARKING else "color:#555"

        # Best iteration por horizonte (promedio)
        if not metrics_df.empty:
            park_metrics = metrics_df[metrics_df["parking_id"] == pid]
            best_iters = park_metrics["best_iteration"].dropna()
            best_iter_str = f"{best_iters.mean():.0f}" if len(best_iters) else "—"
        else:
            best_iter_str = "—"

        # Nº de features y detección de features nuevas
        feat_json = MODELS_DIR / f"{pid}_feature_cols.json"
        n_features = "—"
        has_delta  = "—"
        has_verano = "—"
        if feat_json.exists():
            with open(feat_json, encoding="utf-8") as f:
                feats = json.load(f)
            n_features = len(feats)
            has_delta  = "✅" if "ocu_delta_1" in feats else "❌"
            has_verano = "✅" if "mes_verano"  in feats else "❌"

        rows_html += f"""
        <tr>
          <td><strong>{pid}</strong></td>
          <td>{nombre}</td>
          <td style="{alpha_style}">{alpha:.2f}{'  ★' if pid in ALPHA_POR_PARKING else ''}</td>
          <td>{umbral}</td>
          <td>{n_est}</td>
          <td>{depth}</td>
          <td>{lr}</td>
          <td>{subsamp}</td>
          <td>{best_iter_str}</td>
          <td>{n_features}</td>
          <td style="text-align:center">{has_delta}</td>
          <td style="text-align:center">{has_verano}</td>
        </tr>"""

    return f"""
    <table class="metrics-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Parking</th>
          <th>Alpha (α) ★=override</th>
          <th>Umbral pico</th>
          <th>n_estimators</th>
          <th>max_depth</th>
          <th>learning_rate</th>
          <th>subsample</th>
          <th>Best iter (media)</th>
          <th>Nº features</th>
          <th>ocu_delta_1</th>
          <th>mes_verano</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="legend">★ Alpha con override manual en config.py &nbsp;|&nbsp;
    ocu_delta_1 = velocidad de cambio (lag_1 − lag_2) &nbsp;|&nbsp;
    mes_verano = flag temporada alta (jun–sep)</p>"""


# =============================================================================
# MAPA INTERACTIVO
# =============================================================================

def build_map_data() -> dict:
    """
    Carga coordenadas (parking_reference.csv) y predicciones (CSVs de detalle)
    y los combina en un dict JSON-serializable para el mapa Leaflet.

    Estructura de salida:
      { "parkings": { "AN": { name, lat, lon, cap,
                              ts[], ocu[], p15[], p30[], p45[], p60[],
                              sem[], min_sat[] }, ... } }
    """
    ref_path = OUT_DIR / "parking_reference.csv"
    if not ref_path.exists():
        print("⚠️  parking_reference.csv no encontrado — mapa deshabilitado")
        return {"parkings": {}}

    ref = pd.read_csv(ref_path).set_index("parking_id")
    result = {"parkings": {}}

    for pid in ref.index:
        t15_path = VAL_DIR / f"{pid}_t15_detalle.csv"
        if not t15_path.exists():
            continue
        try:
            lat = float(ref.loc[pid, "latitud"])
            lon = float(ref.loc[pid, "longitud"])
            if pd.isna(lat) or pd.isna(lon):
                continue
        except (KeyError, ValueError, TypeError):
            continue

        df15 = pd.read_csv(t15_path)
        if "timestamp" not in df15.columns:
            continue
        df15["timestamp"] = pd.to_datetime(df15["timestamp"])
        df15 = df15.sort_values("timestamp").reset_index(drop=True)

        merged = df15[["timestamp", "y_real", "y_pred"]].rename(
            columns={"y_real": "ocu", "y_pred": "p15"}
        ).copy()
        merged["sem"]   = df15["semaforo"].fillna("verde") if "semaforo"      in df15.columns else "verde"
        merged["slots"] = df15["slots_hasta_sat"]           if "slots_hasta_sat" in df15.columns else -1.0

        # Predicciones y semáforo de los otros horizontes
        for h, pcol, scol, slcol in [
            ("t30", "p30", "s30", "sl30"),
            ("t45", "p45", "s45", "sl45"),
            ("t60", "p60", "s60", "sl60"),
        ]:
            h_path = VAL_DIR / f"{pid}_{h}_detalle.csv"
            if h_path.exists():
                dh = pd.read_csv(h_path)
                dh["timestamp"] = pd.to_datetime(dh["timestamp"])
                usecols  = ["timestamp", "y_pred"]
                ren      = {"y_pred": pcol}
                if "semaforo"      in dh.columns: usecols.append("semaforo");      ren["semaforo"]      = scol
                if "slots_hasta_sat" in dh.columns: usecols.append("slots_hasta_sat"); ren["slots_hasta_sat"] = slcol
                merged = merged.merge(dh[usecols].rename(columns=ren), on="timestamp", how="left")
            # Garantizar columnas aunque no exista el fichero o le falten campos
            for col, default in [(pcol, None), (scol, "verde"), (slcol, -1.0)]:
                if col not in merged.columns:
                    merged[col] = default

        cap  = int(ref.loc[pid, "plazas_totales"])
        name = str(ref.loc[pid, "parking_name"]) if "parking_name" in ref.columns else pid

        def to_int_list(series):
            out = []
            for v in series:
                try:
                    out.append(None if pd.isna(v) else int(round(float(v))))
                except Exception:
                    out.append(None)
            return out

        def slots_to_min(v):
            try:
                fv = float(v)
                return -1 if (pd.isna(fv) or fv < 0) else int(round(fv * 15))
            except Exception:
                return -1

        result["parkings"][pid] = {
            "name"   : name,
            "lat"    : round(lat, 6),
            "lon"    : round(lon, 6),
            "cap"    : cap,
            "ts"     : merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M").tolist(),
            "ocu"    : to_int_list(merged["ocu"]),
            "p15"    : to_int_list(merged["p15"]),
            "p30"    : to_int_list(merged["p30"]) if "p30" in merged.columns else [None] * len(merged),
            "p45"    : to_int_list(merged["p45"]) if "p45" in merged.columns else [None] * len(merged),
            "p60"    : to_int_list(merged["p60"]) if "p60" in merged.columns else [None] * len(merged),
            # Semáforo por horizonte (para círculos del tooltip)
            "sem"    : merged["sem"].fillna("verde").tolist(),          # t15 → color marcador
            "s30"    : merged["s30"].fillna("verde").tolist() if "s30" in merged.columns else ["verde"] * len(merged),
            "s45"    : merged["s45"].fillna("verde").tolist() if "s45" in merged.columns else ["verde"] * len(merged),
            "s60"    : merged["s60"].fillna("verde").tolist() if "s60" in merged.columns else ["verde"] * len(merged),
            # Tiempo hasta saturación por horizonte (en minutos, -1 = sin riesgo)
            "min_sat"  : [slots_to_min(v) for v in merged["slots"]],
            "ms30"     : [slots_to_min(v) for v in merged["sl30"]] if "sl30" in merged.columns else [-1] * len(merged),
            "ms45"     : [slots_to_min(v) for v in merged["sl45"]] if "sl45" in merged.columns else [-1] * len(merged),
            "ms60"     : [slots_to_min(v) for v in merged["sl60"]] if "sl60" in merged.columns else [-1] * len(merged),
        }
        print(f"    Mapa: {pid} cargado — {len(merged)} timestamps")

    # ── Matriz de distancias Haversine entre parkings ─────────────────────────
    # Para cada parking se calcula la lista de vecinos ordenada por distancia (m)
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
        R = 6_371_000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return int(round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))))

    coords = [(pid, d["lat"], d["lon"]) for pid, d in result["parkings"].items()]
    for pid, d in result["parkings"].items():
        neighbours = []
        for other_pid, olat, olon in coords:
            if other_pid == pid:
                continue
            dist = _haversine_m(d["lat"], d["lon"], olat, olon)
            neighbours.append([other_pid, dist])
        neighbours.sort(key=lambda x: x[1])
        d["nearby"] = neighbours

    return result


def make_map_section_html(map_data: dict) -> str:
    """
    Genera el HTML + JS de la sección del mapa interactivo:
      - Banner "EN VIVO" con el timestamp seleccionado
      - Selectores de fecha y hora (pasos de 15 min)
      - Mapa Leaflet con marcadores por parking coloreados por semáforo
      - Tooltip con ocupación actual, predicciones +15/30/45/60 y tiempo hasta saturación
      - Leyenda del semáforo
    """
    map_json = json.dumps(map_data, ensure_ascii=False, separators=(",", ":"))
    map_json = map_json.replace("</", "<\\/")   # evita cierre anticipado de <script>

    # HTML (f-string — variables Python interpoladas aquí)
    html_part = f"""  <div class="section-title">Mapa en Tiempo Real — Parkings Málaga</div>
  <div class="map-card">
    <div class="map-controls">
      <div class="rt-banner">🔴 EN VIVO &nbsp;·&nbsp; <span id="ts-display">Cargando...</span></div>
      <div class="picker-group">
        <label>Fecha:</label><input type="date" id="map-date">
        <label>Hora:</label><select id="map-time" class="time-sel"></select>
        <button class="btn-map-go"    onclick="goToPicker()">Ir</button>
        <button class="btn-map-reset" onclick="showLatest()">↩ Último dato</button>
      </div>
    </div>
    <div id="parking-map"></div>
    <div class="map-legend">
      <span>🟢 Bajo riesgo (&gt;{SATURACION_UMBRAL_AMARILLO} slots)</span>
      <span>🟡 Moderado ({SATURACION_UMBRAL_ROJO}–{SATURACION_UMBRAL_AMARILLO} slots)</span>
      <span>🔴 Alto riesgo (&lt;{SATURACION_UMBRAL_ROJO} slots)</span>
      <span>⚫ Sin datos</span>
      <span class="legend-note">1 slot = 15&nbsp;min</span>
    </div>
  </div>

  <div class="section-title" style="margin-top:28px">Panel de Urgencia — Momento Seleccionado</div>
  <div class="summary-card">
    <p class="legend" style="margin-bottom:10px">
      Parkings ordenados por urgencia · se actualiza automáticamente con el selector de fecha/hora del mapa
    </p>
    <table class="metrics-table" id="urgency-table">
      <thead>
        <tr>
          <th>#</th><th>Parking</th><th>Ocupación actual</th>
          <th>Estado</th><th>Tiempo est. hasta sat. (t15)</th>
          <th>+30 min</th><th>+45 min</th><th>+60 min</th>
          <th>Alternativa más cercana</th>
        </tr>
      </thead>
      <tbody id="urgency-body">
        <tr><td colspan="9" style="text-align:center;color:#888;padding:16px">Cargando...</td></tr>
      </tbody>
    </table>
  </div>
"""
    # JavaScript (raw string — sin f-string para no escapar llaves de JS y Leaflet)
    js_part  = "\n<script>\n(function(){\n"
    js_part += "let curTs='';\n"
    js_part += "const COLORS={verde:'#27ae60',amarillo:'#f39c12',rojo:'#e74c3c',nodata:'#95a5a6'};\n"
    js_part += f"const PEAK_PCT={int(PEAK_THRESHOLD_PCT*100)};\n"   # umbral rojo (config.py)
    js_part += f"const WARN_PCT={int(PEAK_THRESHOLD_PCT*100)-10};\n" # umbral amarillo (PEAK-10%)
    js_part += "const D=" + map_json + ";\n"
    js_part += r"""
// Índice timestamp → posición de array por parking
const idx={}, allTsSet=new Set();
Object.entries(D.parkings).forEach(([pid,p])=>{
  idx[pid]={};
  p.ts.forEach((t,i)=>{ idx[pid][t]=i; allTsSet.add(t); });
});
const sortedTs=[...allTsSet].sort();

// Solo timestamps donde al menos un parking tiene predicción p15 válida
const validTs=sortedTs.filter(ts=>
  Object.entries(D.parkings).some(([pid,p])=>{
    const i=idx[pid]&&idx[pid][ts]!==undefined?idx[pid][ts]:null;
    return i!==null&&p.p15[i]!==null;
  })
);

// Mapear fecha → lista de horas CON predicciones válidas (no todos los timestamps históricos)
const tssByDate={};
validTs.forEach(ts=>{
  const[d,t]=ts.split('T');
  if(!tssByDate[d]) tssByDate[d]=[];
  tssByDate[d].push(t);
});

// Fecha más cercana que tenga datos (para auto-snap cuando la elegida está vacía)
function nearestDateWithData(requestedDate){
  const allDates=Object.keys(tssByDate).sort();
  if(!allDates.length) return null;
  const rd=new Date(requestedDate).getTime();
  let best=allDates[0], bestDiff=Infinity;
  for(const dd of allDates){
    const diff=Math.abs(new Date(dd).getTime()-rd);
    if(diff<bestDiff){bestDiff=diff;best=dd;}
  }
  return best;
}

// Rellenar el select de hora con los timestamps reales de la fecha elegida.
// Si la fecha no tiene datos, salta automáticamente a la más cercana con datos.
function populateTimeSelect(requestedDate){
  const sel=document.getElementById('map-time');
  let date=requestedDate;
  if(!tssByDate[date]||tssByDate[date].length===0){
    const nearest=nearestDateWithData(requestedDate);
    if(!nearest){sel.innerHTML='';return;}
    date=nearest;
    document.getElementById('map-date').value=date;
  }
  const times=tssByDate[date];
  sel.innerHTML=times.map(t=>`<option value="${t}">${t}</option>`).join('');
}

// Restringir el date picker al rango real de datos y conectar onchange
(function(){
  const dp=document.getElementById('map-date');
  const tsRef=validTs.length?validTs:sortedTs;
  dp.min=tsRef[0].split('T')[0];
  dp.max=tsRef[tsRef.length-1].split('T')[0];
  dp.addEventListener('change',function(){ populateTimeSelect(this.value); });
})();

// Mapa Leaflet — CartoDB Positron (limpio, sin ruido visual)
const lmap=L.map('parking-map').setView([36.7213,-4.4214],14);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{
  attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
  subdomains:'abcd', maxZoom:19
}).addTo(lmap);

// Marcadores por parking
const mkr={};
Object.entries(D.parkings).forEach(([pid,p])=>{
  const m=L.circleMarker([p.lat,p.lon],{
    radius:12, fillColor:COLORS.nodata, color:'#fff', weight:2.5, fillOpacity:0.9
  });
  m.bindTooltip('',{permanent:false,direction:'top',opacity:1,className:'park-tt'});
  m.addTo(lmap);
  mkr[pid]=m;
});

// Helpers
const pct=(v,c)=>c>0?Math.round(v/c*100):0;
const fmt=(v,c)=>v==null?'—':`${v} pz (${pct(v,c)}%)`;
const fmtTs=ts=>{
  const[d,t]=ts.split('T'),[y,m,dd]=d.split('-');
  return `${dd}/${m}/${y} · ${t}`;
};

// Círculo relleno coloreado según el semáforo del horizonte
function dot(sem){
  const c=COLORS[sem]||'#ccc';
  return `<span style="display:inline-block;width:11px;height:11px;border-radius:50%;`
       + `background:${c};vertical-align:middle;margin-left:6px"></span>`;
}
// Celda de predicción: valor numérico + círculo de semáforo del horizonte
function fmtRow(v,cap,sem){
  return v==null
    ? `<span style="color:#ccc">—</span>${dot('nodata')}`
    : `${fmt(v,cap)} ${dot(sem)}`;
}

// Devuelve el parking más cercano que NO esté en semáforo rojo.
// Si todos son rojos, devuelve el más cercano igualmente (fallback).
function findAlternative(pid){
  const p=D.parkings[pid];
  if(!p||!p.nearby||p.nearby.length===0) return null;
  let fallback=null;
  for(const [npid,dist] of p.nearby){
    const np=D.parkings[npid];
    if(!np) continue;
    const ni=curTs&&idx[npid]&&idx[npid][curTs]!==undefined?idx[npid][curTs]:null;
    const sem=ni!==null?(np.sem[ni]||'verde'):'nodata';
    if(sem==='nodata') continue;   // sin datos → no es alternativa válida
    const ocu=ni!==null?np.ocu[ni]:null;
    const entry={pid:npid,name:np.name,dist,ocu,cap:np.cap,sem};
    if(sem!=='rojo') return entry;
    if(fallback===null) fallback=entry;  // todos rojos → devolver el más cercano con dato
  }
  return fallback;
}

function buildTooltip(pid,p,i){
  const sem=p.sem[i]||'verde';           // semáforo t15 → color marcador
  const sc =COLORS[sem]||COLORS.verde;
  const ms =p.min_sat[i];
  const satStr=(ms==null||ms<0)?'Sin riesgo':`${ms} min`;
  const icon=sem==='rojo'?'🔴':sem==='amarillo'?'🟡':'🟢';
  const ocu=p.ocu[i];
  const occStr=ocu!=null?`${ocu} pz (${pct(ocu,p.cap)}%) ${icon}`:'—';
  // Sugerencia de alternativa para estados rojo/amarillo
  let altRow='';
  if(sem==='rojo'||sem==='amarillo'){
    const alt=findAlternative(pid);
    if(alt){
      const distStr=alt.dist>=1000?(alt.dist/1000).toFixed(1)+' km':alt.dist+' m';
      const libres=alt.ocu!=null?alt.cap-alt.ocu:null;
      const altPct=libres!=null&&alt.cap>0?Math.round(libres/alt.cap*100):null;
      const libreStr=libres!=null?`${libres} libres (${altPct}% libre)`:'—';
      const altIcon=alt.sem==='rojo'?'🔴':alt.sem==='amarillo'?'🟡':alt.sem==='verde'?'🟢':'⚫';
      altRow=`<tr class="tt-sep"><td colspan="2"></td></tr>
      <tr style="background:#f0f9f0">
        <td style="color:#666;font-size:11px;white-space:nowrap">Alternativa más cercana</td>
        <td style="font-weight:600">${altIcon} ${alt.name} (${alt.pid})<br>
          <span style="font-weight:400;color:#555;font-size:11px">${distStr} &nbsp;·&nbsp; ${libreStr}</span>
        </td>
      </tr>`;
    }
  }
  return `<div class="tt-wrap">
    <div class="tt-head">🅿️ ${p.name} (${pid})</div>
    <table class="tt-tbl">
      <tr class="tt-dim"><td>Capacidad</td><td>${p.cap} pz</td></tr>
      <tr class="tt-main"><td>Ocupación actual</td><td>${occStr}</td></tr>
      <tr class="tt-sep"><td colspan="2"></td></tr>
      <tr><td>+15 min</td><td>${fmtRow(p.p15[i],p.cap,p.sem[i])}</td></tr>
      <tr><td>+30 min</td><td>${fmtRow(p.p30[i],p.cap,p.s30[i])}</td></tr>
      <tr><td>+45 min</td><td>${fmtRow(p.p45[i],p.cap,p.s45[i])}</td></tr>
      <tr><td>+60 min</td><td>${fmtRow(p.p60[i],p.cap,p.s60[i])}</td></tr>
      <tr class="tt-sep"><td colspan="2"></td></tr>
      <tr style="color:${sc};font-weight:600">
        <td>Tiempo est. hasta sat.</td><td>${satStr}</td>
      </tr>
      ${altRow}
    </table>
  </div>`;
}

// Panel de urgencia: ranking de parkings por velocidad de saturación
function updateRankingPanel(ts){
  const SEM_ORDER={rojo:0,amarillo:1,verde:2,nodata:3};
  const rows=Object.entries(D.parkings).map(([pid,p])=>{
    const i=idx[pid]&&idx[pid][ts]!==undefined?idx[pid][ts]:null;
    if(i===null) return {pid,name:p.name,cap:p.cap,ocu:null,pct_ocu:null,sem:'nodata',ms:-1,ms30:-1,ms45:-1,ms60:-1};
    return {
      pid, name:p.name, cap:p.cap,
      ocu:p.ocu[i], pct_ocu:p.ocu[i]!=null?pct(p.ocu[i],p.cap):null,
      sem:p.sem[i]||'verde',
      ms:p.min_sat[i]??-1, ms30:p.ms30[i]??-1, ms45:p.ms45[i]??-1, ms60:p.ms60[i]??-1,
    };
  });
  rows.sort((a,b)=>{
    const so=SEM_ORDER[a.sem]-SEM_ORDER[b.sem];
    if(so!==0) return so;
    const am=a.ms<0?Infinity:a.ms, bm=b.ms<0?Infinity:b.ms;
    return am-bm;
  });
  const satCell=(ms,sem)=>{
    if(ms<0) return '<span style="color:#27ae60">Sin riesgo</span>';
    const c=COLORS[sem]||'#888';
    return `<span style="color:${c};font-weight:${sem==='rojo'?700:400}">${ms} min</span>`;
  };
  const altCell=(r)=>{
    // Solo se muestra alternativa si el estado es rojo o amarillo
    if(r.sem==='verde'||r.sem==='nodata') return '<span style="color:#ccc">—</span>';
    const alt=findAlternative(r.pid);
    if(!alt) return '<span style="color:#ccc">—</span>';
    const distStr=alt.dist>=1000?(alt.dist/1000).toFixed(1)+' km':alt.dist+' m';
    const libres=alt.ocu!=null?alt.cap-alt.ocu:null;
    const libreStr=libres!=null?libres+' libres':'—';
    const altIcon=alt.sem==='rojo'?'🔴':alt.sem==='amarillo'?'🟡':alt.sem==='verde'?'🟢':'⚫';
    return `${altIcon} <strong>${alt.pid}</strong> · ${distStr} · ${libreStr}`;
  };
  document.getElementById('urgency-body').innerHTML=rows.map((r,n)=>`
    <tr>
      <td>${n+1}</td>
      <td><strong>${r.pid}</strong> · ${r.name}</td>
      <td>${r.ocu!=null?r.ocu+' pz ('+r.pct_ocu+'%)':'—'}</td>
      <td style="text-align:center">${dot(r.sem)}</td>
      <td>${satCell(r.ms,r.sem)}</td>
      <td>${satCell(r.ms30,r.sem)}</td>
      <td>${satCell(r.ms45,r.sem)}</td>
      <td>${satCell(r.ms60,r.sem)}</td>
      <td style="font-size:12px">${altCell(r)}</td>
    </tr>`).join('');
}

// Búsqueda binaria del timestamp con predicción válida más cercano al seleccionado
function findNearestTs(target){
  const arr=validTs.length?validTs:sortedTs;
  const tgt=new Date(target).getTime();
  let lo=0, hi=arr.length-1;
  while(lo<hi){ const mid=(lo+hi)>>1; new Date(arr[mid]).getTime()<tgt?lo=mid+1:hi=mid; }
  if(lo>0){
    const dp=tgt-new Date(arr[lo-1]).getTime();
    const dn=new Date(arr[lo]).getTime()-tgt;
    if(dp<dn) return arr[lo-1];
  }
  return arr[Math.min(lo,arr.length-1)];
}

function syncPicker(ts){
  const[d,t]=ts.split('T');
  document.getElementById('map-date').value=d;
  populateTimeSelect(d);
  document.getElementById('map-time').value=t;
}

function updateMap(ts){
  curTs=ts;
  document.getElementById('ts-display').textContent=fmtTs(ts);
  Object.entries(D.parkings).forEach(([pid,p])=>{
    const m=mkr[pid];
    const i=idx[pid]&&idx[pid][ts]!==undefined?idx[pid][ts]:null;
    if(i===null){
      m.setStyle({fillColor:COLORS.nodata});
      m.setTooltipContent(`<div class="tt-nodata"><b>${p.name}</b><br><i>Sin datos para este momento</i></div>`);
    } else {
      m.setStyle({fillColor:COLORS[p.sem[i]]||COLORS.nodata});
      m.setTooltipContent(buildTooltip(pid,p,i));
    }
  });
  updateRankingPanel(ts);
}

window.showLatest=function(){
  // Busca el último ts donde los 4 horizontes tienen predicción válida.
  // Fallback progresivo: si no hay con 4, acepta 3→2→1 (p15 solo).
  const horizons=["p15","p30","p45","p60"];
  for(let need=horizons.length;need>=1;need--){
    for(let i=sortedTs.length-1;i>=0;i--){
      const t=sortedTs[i];
      const valid=Object.entries(D.parkings).some(([pid,p])=>{
        const ii=idx[pid]&&idx[pid][t]!==undefined?idx[pid][t]:null;
        if(ii===null) return false;
        return horizons.slice(0,need).every(h=>p[h]&&p[h][ii]!==null);
      });
      if(valid){syncPicker(t);updateMap(t);return;}
    }
  }
};
window.goToPicker=function(){
  const d=document.getElementById('map-date').value;
  const t=document.getElementById('map-time').value;
  if(!d||!t) return;
  const ts=findNearestTs(d+'T'+t);
  syncPicker(ts); updateMap(ts);
};

showLatest();
"""
    js_part += "})();\n</script>\n"
    return html_part + js_part


# =============================================================================
# MAIN
# =============================================================================

def build_report():
    print("=" * 70)
    print("GENERANDO REPORTE HTML CONSOLIDADO")
    print("=" * 70)

    # Cargar nombres de parking
    ref_path = OUT_DIR / "parking_reference.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        for _, row in ref.iterrows():
            PARKING_NAMES[row["parking_id"]] = row["parking_name"]

    # Cargar resumen global
    resumen_path = VAL_DIR / "resumen_global.csv"
    if not resumen_path.exists():
        print("⚠️  No se encontró resumen_global.csv — ejecuta primero 03_validate.py")
        resumen = pd.DataFrame()
    else:
        resumen = pd.read_csv(resumen_path)

    # Cargar métricas de entrenamiento (para best_iteration)
    train_metrics_path = OUT_DIR / "metrics_entrenamiento.csv"
    train_metrics = pd.read_csv(train_metrics_path) if train_metrics_path.exists() else pd.DataFrame()

    # Parkings con datos procesados
    pkl_files = sorted(PROC_DIR.glob("*.pkl"))
    parking_ids = [f.stem for f in pkl_files if (MODELS_DIR / f"{f.stem}_t15.joblib").exists()]
    print(f"Parkings con modelos: {parking_ids}")

    sections_html = ""

    for pid in parking_ids:
        nombre = PARKING_NAMES.get(pid, pid)
        print(f"  Procesando {pid} — {nombre}...")

        # Cargar datos de detalle y SHAP por horizonte
        detalle_dfs = {}
        shap_dfs    = {}
        capacidad   = 0

        for h in HORIZONS:
            det_path  = VAL_DIR / f"{pid}_{h}_detalle.csv"
            shap_path = VAL_DIR / "shap" / f"{pid}_{h}_shap.csv"

            if det_path.exists():
                df = pd.read_csv(det_path)
                detalle_dfs[h] = df
                # inferir capacidad desde el primer registro completo
                if "y_real" in df.columns and capacidad == 0:
                    pkl = pd.read_pickle(PROC_DIR / f"{pid}.pkl")
                    capacidad = pkl["capacidad"].iloc[0]

            if shap_path.exists():
                shap_dfs[h] = pd.read_csv(shap_path)

        if not detalle_dfs:
            continue

        # Una figura por horizonte
        images = {}
        for h in HORIZONS:
            if h in detalle_dfs:
                images[h] = make_horizon_figure(
                    pid, h, detalle_dfs[h],
                    shap_dfs.get(h), capacidad
                )

        tabs_html    = make_tabs_html(pid, images)
        metrics_html = make_metrics_table_html(pid, resumen) if not resumen.empty else ""

        # Badge del semáforo: último estado del test en t15
        semaforo_badge = ""
        if "t15" in detalle_dfs and "semaforo" in detalle_dfs["t15"].columns:
            ultimo_estado = detalle_dfs["t15"]["semaforo"].dropna().iloc[-1] if not detalle_dfs["t15"]["semaforo"].dropna().empty else None
            icono_map = {"verde": ("🟢", "#27ae60", "Bajo riesgo"), "amarillo": ("🟡", "#f39c12", "Riesgo moderado"), "rojo": ("🔴", "#e74c3c", "Alto riesgo")}
            if ultimo_estado in icono_map:
                icono, color, etiqueta = icono_map[ultimo_estado]
                semaforo_badge = f'<span class="semaforo-badge" style="border-color:{color}; color:{color}" title="Último slot del test · slots_hasta_sat basado en delta pred_t15">{icono} {etiqueta}</span>'

        sections_html += f"""
        <div class="parking-card" id="{pid}">
          <div class="parking-header">
            <span class="pid-badge">{pid}</span>
            <span class="parking-title">{nombre}</span>
            {semaforo_badge}
            <span class="capacity-badge">Capacidad: {int(capacidad)} plazas</span>
          </div>
          {metrics_html}
          {tabs_html}
        </div>"""

    # Tabla global de métricas
    global_table = make_global_table_html(resumen) if not resumen.empty else "<p><em>Sin datos de resumen global.</em></p>"

    # Tabla de hiperparámetros
    hyperparams_table = make_hyperparams_table_html(parking_ids, train_metrics)

    # Mapa interactivo
    print("  Construyendo datos del mapa...")
    map_data = build_map_data()
    map_html = make_map_section_html(map_data)

    # Índice de parkings
    index_links = " &nbsp;|&nbsp; ".join(
        f'<a href="#{pid}">{pid}</a>' for pid in parking_ids
    )

    # ── HTML completo ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Reporte Parkings Málaga</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #f4f6f8; margin: 0; padding: 0; color: #2c3e50; }}
  .topbar {{ background: #2c3e50; color: white; padding: 18px 32px; position: sticky; top: 0; z-index: 100;
             display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
  .topbar h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
  .topbar .nav {{ font-size: 13px; }}
  .topbar .nav a {{ color: #ecf0f1; margin: 0 6px; text-decoration: none; }}
  .topbar .nav a:hover {{ text-decoration: underline; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}
  .section-title {{ font-size: 20px; font-weight: 700; border-left: 4px solid #3498db;
                    padding-left: 12px; margin: 32px 0 14px; color: #2c3e50; }}
  .parking-card {{ background: white; border-radius: 10px; padding: 24px 28px;
                   margin-bottom: 36px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                   border-top: 4px solid #3498db; }}
  .parking-header {{ display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }}
  .pid-badge {{ background: #2c3e50; color: white; padding: 4px 12px; border-radius: 20px;
                font-size: 15px; font-weight: 700; }}
  .parking-title {{ font-size: 18px; font-weight: 600; }}
  .capacity-badge {{ background: #ecf0f1; padding: 4px 10px; border-radius: 12px;
                     font-size: 13px; color: #555; margin-left: auto; }}
  .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 4px; }}
  .metrics-table th {{ background: #2c3e50; color: white; padding: 9px 12px; text-align: left; font-weight: 600; }}
  .metrics-table td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; }}
  .metrics-table tr:hover td {{ background: #f8f9fa; }}
  .legend {{ font-size: 12px; color: #888; margin: 6px 0 14px; }}
  .legend span {{ display: inline-block; width: 12px; height: 12px;
                  border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
  .summary-card {{ background: white; border-radius: 10px; padding: 20px 28px;
                   box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 28px; }}
  .note {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 10px 16px;
           border-radius: 4px; font-size: 13px; margin-bottom: 18px; }}
  /* ── Tabs de horizonte ── */
  .tab-container {{ margin-top: 18px; }}
  .tab-bar {{ display: flex; gap: 6px; margin-bottom: 12px; }}
  .tab-btn {{
    padding: 7px 22px; border: 2px solid #3498db; border-radius: 20px;
    background: white; color: #3498db; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all 0.18s;
  }}
  .tab-btn:hover  {{ background: #ebf5fb; }}
  .tab-btn.active {{ background: #3498db; color: white; }}
  .tab-panel img  {{ border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.1); }}
  /* ── Semáforo de saturación ── */
  .semaforo-badge {{
    padding: 4px 12px; border-radius: 20px; border: 2px solid;
    font-size: 13px; font-weight: 600; background: white;
  }}
  /* ── Mapa interactivo ── */
  .map-card {{ background: white; border-radius: 10px; overflow: hidden;
               box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 28px; }}
  .map-controls {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
                   padding: 14px 22px; background: #f8f9fa;
                   border-bottom: 1px solid #e8e8e8; }}
  .rt-banner {{ background: #2c3e50; color: white; padding: 6px 16px;
                border-radius: 20px; font-size: 13px; font-weight: 600;
                letter-spacing: 0.3px; }}
  .picker-group {{ display: flex; align-items: center; gap: 8px;
                   font-size: 13px; flex-wrap: wrap; }}
  .picker-group label {{ color: #666; }}
  .picker-group input {{ padding: 5px 8px; border: 1px solid #ddd;
                         border-radius: 6px; font-size: 13px; color: #2c3e50; }}
  .time-sel {{ padding: 5px 8px; border: 1px solid #ddd; border-radius: 6px;
               font-size: 13px; color: #2c3e50; min-width: 90px; cursor: pointer; }}
  .btn-map-go {{ padding: 5px 16px; background: #3498db; color: white; border: none;
                 border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }}
  .btn-map-go:hover {{ background: #2980b9; }}
  .btn-map-reset {{ padding: 5px 16px; background: white; color: #2c3e50;
                    border: 1px solid #ddd; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  .btn-map-reset:hover {{ background: #ecf0f1; }}
  #parking-map {{ height: 520px; }}
  .map-legend {{ display: flex; gap: 18px; flex-wrap: wrap; padding: 10px 22px;
                 background: #f8f9fa; border-top: 1px solid #e8e8e8;
                 font-size: 12px; color: #666; }}
  .legend-note {{ margin-left: auto; color: #aaa; }}
  /* Leaflet tooltip overrides */
  .park-tt {{ padding: 0 !important; border: 1px solid #ddd !important;
              box-shadow: 0 4px 18px rgba(0,0,0,0.15) !important;
              border-radius: 6px !important; }}
  .park-tt.leaflet-tooltip-top::before {{ border-top-color: #2c3e50 !important; }}
  .tt-wrap {{ font-family: 'Segoe UI', sans-serif; font-size: 12px; min-width: 235px; }}
  .tt-head {{ background: #2c3e50; color: white; padding: 8px 12px;
              font-weight: 700; border-radius: 5px 5px 0 0; }}
  .tt-tbl {{ width: 100%; border-collapse: collapse; }}
  .tt-tbl td {{ padding: 3px 12px; }}
  .tt-tbl td:last-child {{ text-align: right; font-weight: 500; }}
  .tt-dim td {{ color: #999; font-size: 11px; }}
  .tt-main td {{ font-weight: 600; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
  .tt-sep td {{ height: 4px; }}
  .tt-nodata {{ padding: 10px 14px; font-size: 12px; color: #888;
                font-family: 'Segoe UI', sans-serif; }}
</style>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>

<script>
function showTab(pid, horizon, btn) {{
  // Ocultar todos los paneles de este parking
  document.querySelectorAll('[id^="tab-' + pid + '-"]').forEach(function(el) {{
    el.style.display = 'none';
  }});
  // Desactivar todos los botones del grupo
  btn.closest('.tab-bar').querySelectorAll('.tab-btn').forEach(function(b) {{
    b.classList.remove('active');
  }});
  // Mostrar panel seleccionado y activar botón
  document.getElementById('tab-' + pid + '-' + horizon).style.display = 'block';
  btn.classList.add('active');
}}
</script>

<div class="topbar">
  <h1>🅿️ Reporte de Predicción — Parkings Málaga</h1>
  <div class="nav">{index_links}</div>
</div>

<div class="container">

  <div class="note">
    <strong>Modelo:</strong> XGBoost Quantile Loss (α={0.70}) &nbsp;|&nbsp;
    <strong>Target:</strong> Plazas ocupadas (absolutas) &nbsp;|&nbsp;
    <strong>Horizontes:</strong> 15, 30, 45 y 60 minutos &nbsp;|&nbsp;
    <strong>Umbral de pico:</strong> {int(PEAK_THRESHOLD_PCT*100)}% de capacidad &nbsp;|&nbsp;
    <strong>Split:</strong> 70% train / 15% valid / 15% test
  </div>

  {map_html}

  <div class="section-title">Resumen Global — Todos los Parkings</div>
  <div class="summary-card">
    <div class="legend">
      <span style="background:#27ae60"></span> Recall ≥ 75% &nbsp;
      <span style="background:#f39c12"></span> Recall 50–75% &nbsp;
      <span style="background:#e74c3c"></span> Recall &lt; 50%
    </div>
    {global_table}
  </div>

  <div class="section-title">Hiperparámetros por Parking</div>
  <div class="summary-card">
    {hyperparams_table}
  </div>

  <div class="section-title">Detalle por Parking</div>
  {sections_html}

</div>
</body>
</html>"""

    out_path = OUT_DIR / "reporte_parkings.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Reporte generado: {out_path}")
    print("   Ábrelo en cualquier navegador (doble clic en el archivo).")


if __name__ == "__main__":
    build_report()
