"""spherex_core.py - Nucleo de la web "SPHEREx Cronologia".

Generaliza los notebooks de M101 (objeto fijo) y de la estrella de Barnard
(objeto con movimiento propio): dado un objeto o unas coordenadas,

  1. resuelve la posicion (y el movimiento propio si lo hay),
  2. busca en IRSA las observaciones SPHEREx (SIA) que cubren esa posicion,
  3. descarga recortes FITS repartidos en el tiempo,
  4. los alinea por coordenadas del cielo y genera un GIF (blink),
  5. devuelve un informe con los datos de cada observacion.

Es una herramienta de INSPECCION VISUAL: no iguala PSF, fondos, respuesta
espectral ni calibraciones relativas. Cada imagen SPHEREx esta en una
longitud de onda distinta, asi que un cambio de brillo entre cuadros puede
ser espectral y no temporal.

Metodo WCS basado en el tutorial de IRSA:
https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts/
"""
from __future__ import annotations

import csv
import io
import re
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")  # servidor sin pantalla
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import numpy as np  # noqa: E402
import requests  # noqa: E402
import astropy.units as u  # noqa: E402
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans  # noqa: E402
from astropy.coordinates import Distance, SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.visualization import AsinhStretch, ImageNormalize  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from astroquery.ipac.irsa import Irsa  # noqa: E402
from reproject import reproject_interp  # noqa: E402

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
# Colecciones SPHEREx en IRSA. Si aparecen nuevas (otro "quick release"),
# anadelas aqui y apareceran como opcion en la web.
COLECCIONES = ["spherex_dr1", "spherex_qr3"]
DETECTORES = ["D1", "D2", "D3", "D4", "D5", "D6"]
PIXEL_SPHEREX_ARCSEC = 6.2

CACHE_DIR = Path("cache")

# Objetos de los notebooks (parametros exactos que ya usaste). Funcionan sin
# depender de SIMBAD. Clave = nombre normalizado (minusculas, sin simbolos).
PRESETS = {
    "m101": dict(nombre="M101", ra=210.80227, dec=54.34895),
    "messier101": dict(nombre="M101", ra=210.80227, dec=54.34895),
    "ngc5457": dict(nombre="M101", ra=210.80227, dec=54.34895),
    "barnard": dict(nombre="Estrella de Barnard", ra=269.452075, dec=4.693391,
                    pm_ra=-802.803, pm_dec=10362.394, parallax=546.976, rv=-110.6),
    "barnardsstar": dict(nombre="Estrella de Barnard", ra=269.452075, dec=4.693391,
                         pm_ra=-802.803, pm_dec=10362.394, parallax=546.976, rv=-110.6),
    "estrelladebarnard": dict(nombre="Estrella de Barnard", ra=269.452075, dec=4.693391,
                              pm_ra=-802.803, pm_dec=10362.394, parallax=546.976, rv=-110.6),
    "gj699": dict(nombre="Estrella de Barnard", ra=269.452075, dec=4.693391,
                  pm_ra=-802.803, pm_dec=10362.394, parallax=546.976, rv=-110.6),
}

_RENDER_LOCK = threading.Lock()  # pyplot no es thread-safe

ProgressFn = Callable[[int, str], None]


def _noop(_pct: int, _msg: str) -> None:
    pass


class ErrorSPHEREx(Exception):
    """Error esperable (sin datos, objeto no encontrado...) con mensaje para el usuario."""


# ----------------------------------------------------------------------------
# Objeto: posicion y movimiento propio
# ----------------------------------------------------------------------------
@dataclass
class Objeto:
    nombre: str
    ra: float                       # grados, ICRS, epoca J2000
    dec: float
    pm_ra: float = 0.0              # mas/ano (ya incluye cos(dec))
    pm_dec: float = 0.0
    parallax: Optional[float] = None  # mas
    rv: Optional[float] = None        # km/s
    fuente: str = ""

    @property
    def pm_total_arcsec_ano(self) -> float:
        return float(np.hypot(self.pm_ra, self.pm_dec)) / 1000.0

    def posicion_en(self, t: Time) -> tuple[float, float]:
        """(RA, Dec) en grados en la fecha `t`, aplicando movimiento propio."""
        if self.pm_total_arcsec_ano == 0:
            return self.ra, self.dec

        if self.parallax and self.parallax > 0:
            c = SkyCoord(
                ra=self.ra * u.deg, dec=self.dec * u.deg,
                pm_ra_cosdec=self.pm_ra * u.mas / u.yr,
                pm_dec=self.pm_dec * u.mas / u.yr,
                distance=Distance(parallax=self.parallax * u.mas),
                radial_velocity=(self.rv or 0.0) * u.km / u.s,
                frame="icrs", obstime=Time("J2000"),
            )
            p = c.apply_space_motion(new_obstime=t)
            return float(p.ra.deg), float(p.dec.deg)

        # Sin paralaje: propagacion lineal (suficiente para ~2 anos).
        dt = t.jyear - 2000.0
        dec = self.dec + self.pm_dec * dt / 3.6e6
        ra = self.ra + self.pm_ra * dt / 3.6e6 / np.cos(np.radians(self.dec))
        return float(ra % 360.0), float(dec)


def _normalizar(texto: str) -> str:
    return re.sub(r"[^a-z0-9]", "", texto.lower())


def _valor(fila, clave) -> Optional[float]:
    """Lee un campo de una fila astropy ignorando enmascarados/NaN."""
    try:
        v = fila[clave]
    except (KeyError, IndexError):
        return None
    if np.ma.is_masked(v) or v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def resolver_objeto(texto: str, pm_ra=None, pm_dec=None, parallax=None, rv=None) -> Objeto:
    """Convierte lo que escribe el usuario en un `Objeto`.

    Acepta: "210.80227 54.34895" (grados), "14:03:12.5 +54:20:56" o
    "14h03m12.5s +54d20m56s" (sexagesimal), o un nombre ("M101", "NGC 1300",
    "Barnard's star", "Proxima Cen"...) resuelto con SIMBAD.
    Los parametros pm_* / parallax / rv, si se dan, reemplazan a los de catalogo.
    """
    texto = (texto or "").strip()
    if not texto:
        raise ErrorSPHEREx("Escribe un nombre o unas coordenadas.")

    base: Optional[Objeto] = None

    # 1) Dos numeros -> grados decimales
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)[\s,;]+([+-]?\d+(?:\.\d+)?)\s*", texto)
    if m:
        ra, dec = float(m.group(1)), float(m.group(2))
        if not (0 <= ra < 360 and -90 <= dec <= 90):
            raise ErrorSPHEREx("Coordenadas fuera de rango (RA 0-360, Dec -90..90).")
        base = Objeto(f"RA {ra:.5f} Dec {dec:+.5f}", ra, dec, fuente="coordenadas dadas")

    # 2) Sexagesimal
    if base is None and (
        ":" in texto
        or re.search(r"\d\s*h\s*\d+\s*m", texto, re.I)
        or re.fullmatch(r"\s*[+-]?\d+\s+\d+\s+[\d.]+[\s,;]+[+-]?\d+\s+\d+\s+[\d.]+\s*", texto)
    ):
        try:
            c = SkyCoord(texto, unit=(u.hourangle, u.deg), frame="icrs")
            base = Objeto(f"RA {c.ra.deg:.5f} Dec {c.dec.deg:+.5f}", float(c.ra.deg),
                          float(c.dec.deg), fuente="coordenadas dadas")
        except Exception as exc:  # noqa: BLE001
            raise ErrorSPHEREx(f"No pude interpretar las coordenadas: {exc}") from exc

    # 3) Presets de tus notebooks
    if base is None and _normalizar(texto) in PRESETS:
        p = PRESETS[_normalizar(texto)]
        base = Objeto(
            p["nombre"], p["ra"], p["dec"],
            pm_ra=p.get("pm_ra", 0.0), pm_dec=p.get("pm_dec", 0.0),
            parallax=p.get("parallax"), rv=p.get("rv"),
            fuente="preset (notebook)",
        )

    # 4) Nombre -> SIMBAD (con Sesame como respaldo)
    if base is None:
        base = _resolver_por_nombre(texto)

    # Reemplazos manuales
    if pm_ra is not None:
        base.pm_ra = float(pm_ra)
    if pm_dec is not None:
        base.pm_dec = float(pm_dec)
    if parallax is not None:
        base.parallax = float(parallax)
    if rv is not None:
        base.rv = float(rv)
    if any(v is not None for v in (pm_ra, pm_dec, parallax, rv)):
        base.fuente += " + parametros manuales"
    return base


def _resolver_por_nombre(nombre: str) -> Objeto:
    try:
        from astroquery.simbad import Simbad

        s = Simbad()
        s.add_votable_fields("pmra", "pmdec", "plx_value", "rvz_radvel")
        tabla = s.query_object(nombre)
        if tabla is not None and len(tabla) > 0:
            fila = tabla[0]
            ra, dec = _valor(fila, "ra"), _valor(fila, "dec")
            if ra is not None and dec is not None:
                ident = str(fila["main_id"]).strip() if "main_id" in tabla.colnames else nombre
                return Objeto(
                    ident or nombre, ra, dec,
                    pm_ra=_valor(fila, "pmra") or 0.0,
                    pm_dec=_valor(fila, "pmdec") or 0.0,
                    parallax=_valor(fila, "plx_value"),
                    rv=_valor(fila, "rvz_radvel"),
                    fuente="SIMBAD",
                )
    except Exception:  # noqa: BLE001  (red caida, version de astroquery...)
        pass

    try:
        c = SkyCoord.from_name(nombre)
        return Objeto(nombre, float(c.ra.deg), float(c.dec.deg), fuente="Sesame (sin movimiento propio)")
    except Exception as exc:  # noqa: BLE001
        raise ErrorSPHEREx(
            f"No encontre el objeto \"{nombre}\". Prueba con otro nombre o con coordenadas."
        ) from exc


# ----------------------------------------------------------------------------
# Busqueda y descarga
# ----------------------------------------------------------------------------
def buscar_observaciones(obj: Objeto, detector: str, colecciones: list[str]):
    """Consulta SIA de IRSA y filtra por detector. Devuelve tabla ordenada por t_min."""
    if detector not in DETECTORES:
        raise ErrorSPHEREx(f"Detector no valido: {detector}")
    colecciones = [c for c in colecciones if c in COLECCIONES]
    if not colecciones:
        raise ErrorSPHEREx("Elige al menos una coleccion SPHEREx.")

    centro = SkyCoord(*obj.posicion_en(Time("2026-01-01")), unit="deg", frame="icrs")
    pm = obj.pm_total_arcsec_ano
    radio = 1.0 if pm < 0.1 else 3.0 * pm + 5.0  # arcsec; amplio si se mueve

    try:
        res = Irsa.query_sia(pos=(centro, radio * u.arcsec), collection=colecciones)
    except Exception as exc:  # noqa: BLE001
        raise ErrorSPHEREx(f"Fallo la consulta a IRSA: {exc}") from exc

    if res is None or len(res) == 0:
        raise ErrorSPHEREx("SPHEREx no tiene imagenes publicadas que cubran esa posicion.")

    banda = np.asarray(res["energy_bandpassname"]).astype(str)
    d = res[banda == f"SPHEREx-{detector}"]
    if len(d) < 2:
        raise ErrorSPHEREx(
            f"Solo hay {len(d)} observacion(es) del detector {detector} en esa posicion; "
            "se necesitan al menos 2. Prueba con otro detector o coleccion."
        )
    d.sort("t_min")
    return d


def elegir_indices(d, n: int) -> list[int]:
    """Observaciones mas cercanas a n fechas equiespaciadas (como en el notebook)."""
    t = np.asarray(d["t_min"], dtype=float)
    objetivos = np.linspace(t.min(), t.max(), min(n, len(d)))
    return sorted({int(np.argmin(np.abs(t - x))) for x in objetivos})


def descargar_recorte(fila, obj: Objeto, size_deg: float, detector: str):
    """Descarga (o reutiliza de cache) el recorte FITS centrado en el objeto en esa fecha."""
    obs_id = str(fila["obs_id"])
    t_obs = Time(float(fila["t_min"]), format="mjd")
    ra, dec = obj.posicion_en(t_obs)

    CACHE_DIR.mkdir(exist_ok=True)
    archivo = CACHE_DIR / f"{obs_id}_{detector}_{ra:.5f}_{dec:+.5f}_{size_deg:.4f}.fits"

    if not archivo.exists():
        url = f"{str(fila['access_url'])}?center={ra},{dec}&size={size_deg}"
        r = requests.get(url, timeout=(20, 180))
        r.raise_for_status()
        with fits.open(io.BytesIO(r.content)) as hdul:  # validar que es FITS
            hdul.verify("exception")
            if "IMAGE" not in hdul:
                raise ValueError("el FITS no tiene extension IMAGE")
        archivo.write_bytes(r.content)

    return dict(
        obs_id=obs_id,
        ruta=archivo,
        t_obs=t_obs,
        fecha=t_obs.strftime("%Y-%m-%d"),
        coleccion=str(fila["obs_collection"]).replace("spherex_", "").upper(),
        ra=ra,
        dec=dec,
    )


# ----------------------------------------------------------------------------
# Lectura y procesado de imagenes
# ----------------------------------------------------------------------------
def leer_recorte(ruta, ra, dec):
    """Imagen, WCS espacial y banda espectral en (ra, dec) de un recorte SPHEREx."""
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    with fits.open(ruta) as hdul:
        hdu = hdul["IMAGE"]
        imagen = np.asarray(hdu.data, dtype=float).copy()
        wcs = WCS(hdu.header).celestial
        x, y = wcs.world_to_pixel(coord)
        if not (0 <= x < imagen.shape[1] and 0 <= y < imagen.shape[0]):
            raise ValueError("la posicion no cae dentro del recorte")

        lam = ancho = None
        try:  # WCS-WAVE es alternativo (key "W") y no debe usar la distorsion SIP
            wcs_esp = WCS(hdu.header, fobj=hdul, key="W")
            wcs_esp.sip = None
            lon, anc = wcs_esp.pixel_to_world(x, y)
            lam = float(lon.to_value(u.um))
            ancho = float(anc.to_value(u.um))
        except Exception:  # noqa: BLE001
            pass

        meta = dict(
            x=float(x), y=float(y), lambda_um=lam, ancho_um=ancho,
            unidad=str(hdu.header.get("BUNIT", "sin especificar")),
            version=str(hdul[0].header.get("VERSION", "no indicada")),
        )
    return imagen, wcs, meta


def rellenar_huecos(imagen, x, y, radio=6, iteraciones=5):
    """Interpola los NaN a menos de `radio` px de (x, y). Solo para visualizar:
    los valores interpolados NO son flujo medido."""
    salida = imagen.copy()
    yy, xx = np.indices(imagen.shape)
    huecos = (np.hypot(xx - x, yy - y) <= radio) & ~np.isfinite(imagen)
    if not huecos.any():
        return salida, huecos
    kernel = Gaussian2DKernel(1.5)
    for _ in range(iteraciones):
        tmp = interpolate_replace_nans(salida, kernel, boundary="extend")
        salida[huecos] = tmp[huecos]
        if not (huecos & ~np.isfinite(salida)).any():
            break
    return salida, huecos


def valor_central(imagen, x, y, radio=2.0):
    """Mediana de los pixeles validos a <= radio px del objeto (imagen original)."""
    yy, xx = np.indices(imagen.shape)
    zona = (np.hypot(xx - x, yy - y) <= radio) & np.isfinite(imagen)
    return float(np.median(imagen[zona])) if zona.any() else None


def centroide(img, x0, y0, r=4, iteraciones=3):
    """Centroide ponderado por flujo, recentrando la ventana en cada paso."""
    yy, xx = np.indices(img.shape)
    fondo = np.nanmedian(img[np.isfinite(img)])
    for _ in range(iteraciones):
        m = (np.hypot(xx - x0, yy - y0) <= r) & np.isfinite(img)
        if not m.any():
            return np.nan, np.nan
        w = np.clip(img[m] - fondo, 0, None)
        if w.sum() == 0:
            return np.nan, np.nan
        x0 = (xx[m] * w).sum() / w.sum()
        y0 = (yy[m] * w).sum() / w.sum()
    return x0, y0


def _fmt_lam(lam) -> str:
    return f"{lam:.2f} um" if lam is not None else "lambda n/d"


# ----------------------------------------------------------------------------
# Pipeline completo
# ----------------------------------------------------------------------------
def generar_cronologia(
    texto: str,
    salida_dir: Path,
    n_imagenes: int = 6,
    detector: str = "D2",
    size_arcmin: float = 6.0,
    colecciones: Optional[list[str]] = None,
    rellenar: bool = True,
    marcar: bool = True,
    fps: int = 2,
    pm_ra=None, pm_dec=None, parallax=None, rv=None,
    progreso: ProgressFn = _noop,
) -> dict:
    colecciones = colecciones or list(COLECCIONES)
    salida_dir = Path(salida_dir)
    salida_dir.mkdir(parents=True, exist_ok=True)
    avisos: list[str] = []

    # 1. Objeto ---------------------------------------------------------------
    progreso(3, "Resolviendo el objeto...")
    obj = resolver_objeto(texto, pm_ra, pm_dec, parallax, rv)
    progreso(8, f"Objeto: {obj.nombre} (RA {obj.ra:.5f}, Dec {obj.dec:+.5f})")

    # 2. Consulta IRSA --------------------------------------------------------
    progreso(10, f"Consultando IRSA ({detector}, {', '.join(colecciones)})...")
    d = buscar_observaciones(obj, detector, colecciones)
    n_total = len(d)
    progreso(20, f"{n_total} observaciones {detector} disponibles.")

    # 3. Seleccion y descarga -------------------------------------------------
    indices = elegir_indices(d, n_imagenes)
    size_deg = size_arcmin / 60.0
    obs = []
    for k, i in enumerate(indices, start=1):
        fila = d[i]
        progreso(20 + int(40 * k / len(indices)),
                 f"Descargando recorte {k}/{len(indices)} ({fila['obs_id']})...")
        try:
            obs.append(descargar_recorte(fila, obj, size_deg, detector))
        except Exception as exc:  # noqa: BLE001
            avisos.append(f"No se pudo descargar {fila['obs_id']}: {exc}")
    if len(obs) < 2:
        raise ErrorSPHEREx("Se descargaron menos de 2 recortes validos; no se puede comparar.")
    obs.sort(key=lambda o: o["t_obs"].mjd)

    # 4. Lectura --------------------------------------------------------------
    progreso(62, "Leyendo imagenes y WCS...")
    datos = []
    for o in obs:
        try:
            im, wcs, meta = leer_recorte(o["ruta"], o["ra"], o["dec"])
        except Exception as exc:  # noqa: BLE001
            avisos.append(f"{o['obs_id']} descartada: {exc}")
            continue
        datos.append(dict(o=o, imagen=im, wcs=wcs, meta=meta,
                          central=valor_central(im, meta["x"], meta["y"])))
    if len(datos) < 2:
        raise ErrorSPHEREx("Menos de 2 imagenes utilizables despues de leer los recortes.")
    unidades = {d_["meta"]["unidad"] for d_ in datos}
    if len(unidades) > 1:
        raise ErrorSPHEREx(f"Unidades BUNIT distintas entre imagenes: {unidades}")
    n = len(datos)

    # 5. Relleno + alineacion -------------------------------------------------
    progreso(68, "Alineando imagenes por coordenadas del cielo...")
    imgs, huecos, n_rell = [], [], []
    for d_ in datos:
        if rellenar:
            im, h = rellenar_huecos(d_["imagen"], d_["meta"]["x"], d_["meta"]["y"])
        else:
            im, h = d_["imagen"], np.zeros(d_["imagen"].shape, bool)
        imgs.append(im)
        huecos.append(h)
        n_rell.append(int(h.sum()))

    wcs_ref = datos[0]["wcs"]
    forma = imgs[0].shape
    alineadas = [imgs[0]]
    huecos_al = [huecos[0]]
    comunes = np.isfinite(imgs[0])
    for im, h, d_ in zip(imgs[1:], huecos[1:], datos[1:]):
        al, huella = reproject_interp((im, d_["wcs"]), wcs_ref, shape_out=forma, order="bilinear")
        hal, _ = reproject_interp((h.astype(float), d_["wcs"]), wcs_ref,
                                  shape_out=forma, order="nearest-neighbor")
        alineadas.append(al)
        huecos_al.append(np.nan_to_num(hal) > 0.5)
        comunes &= np.isfinite(al) & (huella > 0.99)
    if comunes.sum() < 10:
        raise ErrorSPHEREx("No hay suficientes pixeles validos en la cobertura comun.")

    valores = np.concatenate([a[comunes] for a in alineadas])
    vmin, vmax = np.percentile(valores, [1, 99.5])
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
        raise ErrorSPHEREx("No se pudo establecer una escala de intensidad valida.")
    norma = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    mapa = plt.colormaps["magma"].copy()
    mapa.set_bad("black")
    cuadros = [np.ma.masked_where(~comunes, a) for a in alineadas]

    pix = []
    for d_ in datos:
        px, py = wcs_ref.world_to_pixel(
            SkyCoord(ra=d_["o"]["ra"] * u.deg, dec=d_["o"]["dec"] * u.deg, frame="icrs"))
        pix.append((float(px), float(py)))

    etiquetas = [
        f"{obj.nombre} | {d_['o']['coleccion']} {d_['o']['fecha']} | {_fmt_lam(d_['meta']['lambda_um'])}"
        for d_ in datos
    ]

    # 6. Figuras --------------------------------------------------------------
    progreso(78, "Generando el GIF...")
    gif = salida_dir / "cronologia.gif"
    with _RENDER_LOCK:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs_ref)
        vista = ax.imshow(cuadros[0], origin="lower", cmap=mapa, norm=norma, interpolation="nearest")
        marcador = None
        if marcar:
            marcador = ax.scatter([pix[0][0]], [pix[0][1]], marker="o", facecolors="none",
                                  edgecolors="cyan", s=220, linewidths=1.2)
        ax.set_xlabel("Ascension recta (RA)")
        ax.set_ylabel("Declinacion (Dec)")
        titulo = ax.set_title(etiquetas[0], fontsize=10)

        def actualizar(frame):
            vista.set_data(cuadros[frame])
            titulo.set_text(etiquetas[frame])
            if marcador is not None:
                marcador.set_offsets([list(pix[frame])])
            return vista, titulo

        anim = FuncAnimation(fig, actualizar, frames=n, interval=int(1000 / fps), blit=False)
        try:
            anim.save(str(gif), writer=PillowWriter(fps=fps), dpi=110)
        finally:
            plt.close(fig)

        # Rejilla estatica con la misma escala
        progreso(88, "Generando la rejilla comparativa...")
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                                 subplot_kw={"projection": wcs_ref}, squeeze=False)
        axes = axes.ravel()
        im_ = None
        for a, al, d_, (px, py), h in zip(axes, alineadas, datos, pix, huecos_al):
            im_ = a.imshow(np.where(comunes, al, np.nan), origin="lower", cmap=mapa, norm=norma)
            if marcar:
                a.scatter(px, py, marker="o", facecolors="none", edgecolors="cyan",
                          s=220, linewidths=1.2)
            if h.any():
                a.contour(h.astype(float), levels=[0.5], colors="white", linewidths=0.8)
            a.set_title(f"{d_['o']['coleccion']} {d_['o']['fecha']}\n"
                        f"{_fmt_lam(d_['meta']['lambda_um'])}", fontsize=9)
            a.set_xlabel("RA")
            a.set_ylabel("Dec")
        for a in axes[n:]:
            a.axis("off")
        fig.colorbar(im_, ax=list(axes[:n]), shrink=0.8, label=f"Intensidad ({datos[0]['meta']['unidad']})")
        grilla = salida_dir / "rejilla.png"
        fig.savefig(grilla, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 7. Movimiento medido vs esperado (solo si el objeto se mueve) -----------
    movimiento = None
    archivo_mov = None
    ref = SkyCoord(ra=datos[0]["o"]["ra"] * u.deg, dec=datos[0]["o"]["dec"] * u.deg)
    ultimo = SkyCoord(ra=datos[-1]["o"]["ra"] * u.deg, dec=datos[-1]["o"]["dec"] * u.deg)
    desplazamiento = float(ref.separation(ultimo).arcsec)

    if desplazamiento >= 1.0:
        progreso(93, "Midiendo el movimiento del objeto...")
        cents = [centroide(al, px, py) for al, (px, py) in zip(alineadas, pix)]
        ok = [i for i, c in enumerate(cents) if np.all(np.isfinite(c))]
        if len(ok) >= 3:
            def offsets(ra, dec):
                return ((ra - ref.ra.deg) * np.cos(np.radians(ref.dec.deg)) * 3600,
                        (dec - ref.dec.deg) * 3600)

            sky = [wcs_ref.pixel_to_world(*cents[i]) for i in ok]
            medidos = np.array([offsets(c.ra.deg, c.dec.deg) for c in sky])
            esperados = np.array([offsets(datos[i]["o"]["ra"], datos[i]["o"]["dec"]) for i in ok])
            t = np.array([datos[i]["o"]["t_obs"].jyear for i in ok])
            if np.ptp(t) > 0:
                pm_med_ra = float(np.polyfit(t - t[0], medidos[:, 0], 1)[0] * 1000)
                pm_med_dec = float(np.polyfit(t - t[0], medidos[:, 1], 1)[0] * 1000)
                movimiento = dict(
                    medido_ra_mas_ano=pm_med_ra, medido_dec_mas_ano=pm_med_dec,
                    catalogo_ra_mas_ano=obj.pm_ra, catalogo_dec_mas_ano=obj.pm_dec,
                )
                with _RENDER_LOCK:
                    fig2, axs = plt.subplots(1, 2, figsize=(10, 4))
                    for a, k, nom in zip(axs, (0, 1), ("RA*cos(dec)", "Dec")):
                        a.plot(t, esperados[:, k], "-", color="gray", label="esperado (catalogo)")
                        a.plot(t, medidos[:, k], "o", color="tab:orange", label="medido (centroide)")
                        a.set_xlabel("Fecha (ano)")
                        a.set_ylabel(f"Desplazamiento en {nom} (arcsec)")
                        a.grid(alpha=0.3)
                        a.legend()
                    fig2.tight_layout()
                    archivo_mov = salida_dir / "movimiento.png"
                    fig2.savefig(archivo_mov, dpi=130, bbox_inches="tight")
                    plt.close(fig2)
        else:
            avisos.append("No hay suficientes centroides para medir el movimiento propio.")

    # 8. Avisos y tabla -------------------------------------------------------
    lams = [d_["meta"]["lambda_um"] for d_ in datos if d_["meta"]["lambda_um"] is not None]
    rango_lam = float(max(lams) - min(lams)) if lams else None
    if rango_lam is not None and rango_lam > 0.05:
        avisos.append("Las imagenes estan en longitudes de onda distintas: el cambio de brillo "
                      "puede ser espectral y no solo temporal o de movimiento.")
    if any(n_rell):
        avisos.append("Se interpolaron pixeles NaN junto al objeto (contorno blanco en la rejilla); "
                      "esos valores no son flujo medido.")
    if desplazamiento >= 1.0:
        avisos.append(f"Desplazamiento esperado entre la primera y la ultima fecha: "
                      f"{desplazamiento:.2f} arcsec (1 pixel SPHEREx ~ {PIXEL_SPHEREX_ARCSEC} arcsec).")
    avisos.append("Demostracion visual exploratoria; no es un detector validado de variabilidad.")

    filas = []
    for i, (d_, (px, py)) in enumerate(zip(datos, pix), start=1):
        o, m = d_["o"], d_["meta"]
        filas.append(dict(
            n=i, obs_id=o["obs_id"], coleccion=o["coleccion"], fecha=o["fecha"],
            mjd=float(o["t_obs"].mjd), lambda_um=m["lambda_um"], ancho_um=m["ancho_um"],
            ra=o["ra"], dec=o["dec"], pix_x=m["x"], pix_y=m["y"],
            valor_central=d_["central"], pix_rellenados=n_rell[i - 1],
        ))

    csv_path = salida_dir / "observaciones.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    progreso(100, "Listo.")
    return dict(
        objeto=asdict(obj),
        parametros=dict(detector=detector, n_pedidas=n_imagenes, size_arcmin=size_arcmin,
                        colecciones=colecciones),
        resumen=dict(
            n_disponibles=n_total, n_usadas=n,
            cobertura_comun_pct=float(comunes.mean() * 100),
            rango_lambda_um=rango_lam, unidad=datos[0]["meta"]["unidad"],
            version=datos[0]["meta"]["version"],
            fecha_inicial=filas[0]["fecha"], fecha_final=filas[-1]["fecha"],
            desplazamiento_arcsec=desplazamiento,
        ),
        observaciones=filas,
        movimiento=movimiento,
        avisos=avisos,
        archivos=dict(
            gif=gif.name, rejilla=grilla.name, csv=csv_path.name,
            movimiento=archivo_mov.name if archivo_mov else None,
        ),
    )
