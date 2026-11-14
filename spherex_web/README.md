# SPHEREx Cronología

Aplicación web que, dado un objeto astronómico (nombre o coordenadas), genera un
**GIF con sus observaciones de [SPHEREx](https://spherex.caltech.edu/) a lo largo del tiempo**
y una tabla con los datos de cada observación. Usa los datos públicos de
[NASA/IPAC IRSA](https://irsa.ipac.caltech.edu/).

Funciona tanto con objetos fijos (galaxias como M101) como con objetos de alto
movimiento propio (estrella de Barnard): la posición se calcula para la fecha exacta
de cada observación.

## Qué produce

- GIF animado (blink) con los recortes alineados por coordenadas del cielo
- Tabla por observación: fecha, `obs_id`, colección, longitud de onda, ancho de banda,
  posición, valor central (descargable en CSV)
- Rejilla comparativa con la misma escala de intensidad
- Para objetos en movimiento: movimiento propio medido vs. catálogo

## Instalación

```bash
git clone https://github.com/<tu-usuario>/spherex-cronologia.git
cd spherex-cronologia
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Abre <http://127.0.0.1:5000>. Requiere conexión a internet (IRSA y SIMBAD).

### Con Docker

```bash
docker build -t spherex-cronologia .
docker run -p 8000:8000 spherex-cronologia
```

## Entrada aceptada

| Formato | Ejemplo |
|---|---|
| Nombre (SIMBAD) | `M101`, `NGC 1300`, `Barnard's star` |
| Grados | `210.80227 54.34895` |
| Sexagesimal | `14:03:12.5 +54:20:56` |

Para estrellas, el movimiento propio, la paralaje y la velocidad radial se toman de
SIMBAD; se pueden sobrescribir en la sección «Movimiento propio». `M101` y `Barnard`
usan parámetros fijos definidos en `PRESETS` (`spherex_core.py`).

## Estructura

```
app.py                 servidor Flask, cola de trabajos y API
spherex_core.py        consulta IRSA, descarga, WCS, alineación y GIF
templates/index.html   interfaz
notebooks/             notebooks originales (M101 y estrella de Barnard)
```

API: `POST /api/analizar` → `{job_id}`; `GET /api/estado/<job_id>` → progreso y resultado.

## Configuración

- Nuevas colecciones SPHEREx: añade su nombre a `COLECCIONES` en `spherex_core.py`.
- Los recortes se guardan en `cache/` y los resultados en `salidas/` (ignorados por git).
- Si lo despliegas en un servidor, usa 1 worker de gunicorn (los trabajos viven en
  memoria) y limpia `salidas/` periódicamente. GitHub Pages no sirve para esto porque
  necesita un backend Python; usa Render, Railway, Fly.io, Hugging Face Spaces, etc.

## Limitaciones

Es una herramienta de **inspección visual**, no un detector validado de variabilidad.
Cada imagen SPHEREx se toma en una longitud de onda distinta y no se igualan PSF,
fondos ni calibraciones relativas, así que un cambio de brillo puede ser espectral y
no temporal. Los píxeles interpolados junto al objeto no son flujo medido.

## Créditos

Datos: SPHEREx (NASA/Caltech) vía NASA/IPAC Infrared Science Archive. Tratamiento WCS
basado en el [tutorial de cutouts de IRSA](https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts/).

## Licencia

MIT (ver `LICENSE`).
