# =============================================================================
# run_pipeline.py — Orquestador del pipeline de predicción de parkings
#
# Ejecuta los 4 pasos en secuencia con una sola línea de comando.
# Los hiperparámetros se pueden sobrescribir desde la terminal sin tocar
# config.py ni los scripts 01-04.
#
# USO BÁSICO
# ----------
#   python run_pipeline.py                        # pipeline completo
#   python run_pipeline.py --steps 2 3            # solo entrenar y validar
#   python run_pipeline.py --skip-prep            # omitir preprocesado (datos ya generados)
#
# OVERRIDES DE HIPERPARÁMETROS
# ----------------------------
#   python run_pipeline.py --alpha 0.75
#   python run_pipeline.py --alpha-override AN:0.85 PA:0.80
#   python run_pipeline.py --peak-threshold 0.80
#   python run_pipeline.py --excluir LI SA
#   python run_pipeline.py --n-estimators 800 --max-depth 7 --learning-rate 0.03
#
# COMBINACIONES
# -------------
#   python run_pipeline.py --steps 2 3 --alpha 0.80 --alpha-override CR:0.75
#   python run_pipeline.py --skip-prep --peak-threshold 0.75 --excluir LI SA
#
# Los overrides se escriben en _config_override.json antes de cada ejecución
# y se eliminan automáticamente al terminar.
# =============================================================================

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE PASOS
# =============================================================================
STEPS = {
    1: {"script": "01_preprocessing.py", "nombre": "Preprocesado"},
    2: {"script": "02_train.py",          "nombre": "Entrenamiento"},
    3: {"script": "03_validate.py",       "nombre": "Validación"},
    4: {"script": "04_report.py",         "nombre": "Reporte HTML"},
}

OVERRIDE_FILE = Path(__file__).parent / "_config_override.json"


# =============================================================================
# ARGUMENTOS
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline de predicción de ocupación de parkings — Málaga TFM",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Selección de pasos ──────────────────────────────────────────────────
    p.add_argument(
        "--steps", nargs="+", type=int, choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        metavar="N",
        help="Pasos a ejecutar (1=prep 2=train 3=validate 4=report). Ej: --steps 2 3",
    )
    p.add_argument(
        "--skip-prep", action="store_true",
        help="Atajo: omite el paso 1 (equivale a --steps 2 3 4)",
    )

    # ── Overrides de alpha ──────────────────────────────────────────────────
    p.add_argument(
        "--alpha", type=float, default=None,
        metavar="A",
        help="Alpha global de quantile loss. Rango [0,1]. Default: usa config.py",
    )
    p.add_argument(
        "--alpha-override", nargs="+", default=None,
        metavar="PID:VAL",
        help="Alpha por parking. Ej: --alpha-override AN:0.85 PA:0.80",
    )

    # ── Umbral de pico ──────────────────────────────────────────────────────
    p.add_argument(
        "--peak-threshold", type=float, default=None,
        metavar="T",
        help="Umbral de pico como fracción de capacidad. Default: usa config.py (0.85)",
    )

    # ── Parkings ────────────────────────────────────────────────────────────
    p.add_argument(
        "--excluir", nargs="+", default=None,
        metavar="PID",
        help="Parkings a excluir además del default (LI). Ej: --excluir SA",
    )

    # ── Hiperparámetros XGBoost ─────────────────────────────────────────────
    p.add_argument("--n-estimators", type=int,   default=None, metavar="N",
                   help="Número de árboles. Default: usa config.py (500)")
    p.add_argument("--max-depth",    type=int,   default=None, metavar="D",
                   help="Profundidad máxima del árbol. Default: usa config.py (6)")
    p.add_argument("--learning-rate", type=float, default=None, metavar="LR",
                   help="Learning rate (eta). Default: usa config.py (0.05)")

    return p.parse_args()


# =============================================================================
# UTILIDADES
# =============================================================================
def fmt_tiempo(segundos: float) -> str:
    if segundos < 60:
        return f"{segundos:.1f}s"
    m, s = divmod(int(segundos), 60)
    return f"{m}m {s:02d}s"


def build_overrides(args) -> dict:
    """Construye el dict de overrides a partir de los argumentos."""
    ov = {}

    if args.alpha is not None:
        ov["ALPHA_GLOBAL"] = args.alpha

    if args.alpha_override:
        parsed = {}
        for item in args.alpha_override:
            try:
                pid, val = item.split(":")
                parsed[pid.upper()] = float(val)
            except ValueError:
                print(f"⚠️  Formato incorrecto en --alpha-override '{item}' (esperado PID:VALOR)")
                sys.exit(1)
        ov["ALPHA_POR_PARKING"] = parsed

    if args.peak_threshold is not None:
        ov["PEAK_THRESHOLD_PCT"] = args.peak_threshold

    if args.excluir is not None:
        # Se añaden a los excluidos por defecto en config.py (LI)
        ov["PARKINGS_EXCLUIR"] = ["LI"] + [p.upper() for p in args.excluir]

    if args.n_estimators is not None:
        ov["N_ESTIMATORS"] = args.n_estimators
    if args.max_depth is not None:
        ov["MAX_DEPTH"] = args.max_depth
    if args.learning_rate is not None:
        ov["LEARNING_RATE"] = args.learning_rate

    return ov


def write_overrides(ov: dict):
    OVERRIDE_FILE.write_text(
        json.dumps(ov, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  📝 Overrides activos: {json.dumps(ov)}")


def clean_overrides():
    if OVERRIDE_FILE.exists():
        OVERRIDE_FILE.unlink()


def run_step(step_id: int) -> bool:
    """Ejecuta un paso. Devuelve True si tuvo éxito."""
    info = STEPS[step_id]
    script = Path(__file__).parent / info["script"]

    if not script.exists():
        print(f"  ❌ No se encuentra el script: {script}")
        return False

    print(f"\n{'─'*70}")
    print(f"  PASO {step_id}: {info['nombre']}  ({info['script']})")
    print(f"{'─'*70}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(script.parent),
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  ✅ Paso {step_id} completado en {fmt_tiempo(elapsed)}")
        return True
    else:
        print(f"\n  ❌ Paso {step_id} terminó con error (código {result.returncode})")
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    # Atajo --skip-prep
    steps = args.steps
    if args.skip_prep:
        steps = [s for s in steps if s != 1]
        if not steps:
            steps = [2, 3, 4]

    steps = sorted(set(steps))

    # Banner
    print("=" * 70)
    print("  PIPELINE DE PREDICCIÓN DE PARKINGS — Málaga TFM")
    print("=" * 70)
    print(f"  Pasos a ejecutar: {steps}")

    # Construir y escribir overrides
    overrides = build_overrides(args)
    if overrides:
        write_overrides(overrides)
    else:
        print("  ℹ️  Sin overrides: usando valores de config.py")

    # Ejecutar pasos
    t_total = time.time()
    resultados = {}

    try:
        for step_id in steps:
            ok = run_step(step_id)
            resultados[step_id] = ok
            if not ok:
                print(f"\n  ⛔ Pipeline detenido en el paso {step_id}.")
                print("     Revisa el error arriba antes de continuar.")
                break
    finally:
        # Siempre limpiar el override file al terminar
        clean_overrides()

    # Resumen final
    elapsed_total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  RESUMEN — Tiempo total: {fmt_tiempo(elapsed_total)}")
    print(f"{'='*70}")
    for step_id, ok in resultados.items():
        estado = "✅" if ok else "❌"
        print(f"  {estado} Paso {step_id}: {STEPS[step_id]['nombre']}")

    pasos_pendientes = [s for s in steps if s not in resultados]
    for step_id in pasos_pendientes:
        print(f"  ⏭️  Paso {step_id}: {STEPS[step_id]['nombre']}  (no ejecutado)")

    todo_ok = all(resultados.values())
    if todo_ok:
        print("\n  🎉 Pipeline completado sin errores.")
    else:
        print("\n  ⚠️  Pipeline completado con errores. Revisa los mensajes arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()
