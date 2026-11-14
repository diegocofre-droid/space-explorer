"""app.py - Servidor web de SPHEREx Cronologia.

Uso:
    pip install -r requirements.txt
    python app.py
    -> abre http://127.0.0.1:5000
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import spherex_core as core

BASE = Path(__file__).parent
SALIDAS = BASE / "salidas"
SALIDAS.mkdir(exist_ok=True)
core.CACHE_DIR = BASE / "cache"

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=2)   # 2 analisis a la vez; el resto espera
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _num(valor, defecto=None, minimo=None, maximo=None, entero=False):
    """Convierte a numero (o None si viene vacio) y limita al rango permitido."""
    if valor is None or str(valor).strip() == "":
        return defecto
    try:
        x = float(str(valor).replace(",", "."))
    except ValueError:
        raise core.ErrorSPHEREx(f"Valor numerico no valido: {valor!r}")
    if minimo is not None:
        x = max(minimo, x)
    if maximo is not None:
        x = min(maximo, x)
    return int(round(x)) if entero else x


def _ejecutar(job_id: str, params: dict) -> None:
    def progreso(pct: int, msg: str) -> None:
        with LOCK:
            job = JOBS[job_id]
            job["progreso"] = pct
            job["log"].append(msg)

    with LOCK:
        JOBS[job_id]["estado"] = "ejecutando"
    try:
        informe = core.generar_cronologia(salida_dir=SALIDAS / job_id, progreso=progreso, **params)
        with LOCK:
            JOBS[job_id].update(estado="listo", resultado=informe, progreso=100)
    except core.ErrorSPHEREx as exc:
        with LOCK:
            JOBS[job_id].update(estado="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        with LOCK:
            JOBS[job_id].update(estado="error", error=f"Error inesperado: {exc}")


@app.get("/")
def index():
    return render_template(
        "index.html", colecciones=core.COLECCIONES, detectores=core.DETECTORES
    )


@app.post("/api/analizar")
def analizar():
    datos = request.get_json(silent=True) or {}
    try:
        texto = str(datos.get("objeto", "")).strip()
        if not texto or len(texto) > 120:
            raise core.ErrorSPHEREx("Escribe un nombre o unas coordenadas (max. 120 caracteres).")
        colecciones = [c for c in datos.get("colecciones", core.COLECCIONES) if c in core.COLECCIONES]
        detector = datos.get("detector", "D2")
        if detector not in core.DETECTORES:
            detector = "D2"
        params = dict(
            texto=texto,
            n_imagenes=_num(datos.get("n_imagenes"), 6, 2, 12, entero=True),
            detector=detector,
            size_arcmin=_num(datos.get("size_arcmin"), 6.0, 1.0, 30.0),
            colecciones=colecciones,
            rellenar=bool(datos.get("rellenar", True)),
            marcar=bool(datos.get("marcar", True)),
            fps=_num(datos.get("fps"), 2, 1, 5, entero=True),
            pm_ra=_num(datos.get("pm_ra")),
            pm_dec=_num(datos.get("pm_dec")),
            parallax=_num(datos.get("parallax")),
            rv=_num(datos.get("rv")),
        )
    except core.ErrorSPHEREx as exc:
        return jsonify(error=str(exc)), 400

    job_id = uuid.uuid4().hex
    with LOCK:
        JOBS[job_id] = dict(estado="en cola", progreso=0, log=[], creado=time.time())
    executor.submit(_ejecutar, job_id, params)
    return jsonify(job_id=job_id)


@app.get("/api/estado/<job_id>")
def estado(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error="Trabajo no encontrado"), 404
        return jsonify(job)


@app.get("/salidas/<job_id>/<path:archivo>")
def salida(job_id: str, archivo: str):
    if len(job_id) != 32 or not all(c in "0123456789abcdef" for c in job_id):
        return jsonify(error="No encontrado"), 404
    return send_from_directory(SALIDAS / job_id, archivo)


if __name__ == "__main__":
    # threaded=True: el navegador puede consultar el estado mientras se procesa
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
