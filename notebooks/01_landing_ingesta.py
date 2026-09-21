# Databricks notebook source
# MAGIC %md
# MAGIC # FEN-RISK Perú — 01. Landing: Ingesta de fuentes crudas
# MAGIC
# MAGIC Descarga las fuentes de datos directo desde sus URLs originales, las deja sin modificar
# MAGIC en el Volume `workspace.fen_risk.landing` (capa **Landing** = RAW preservado tal cual llega),
# MAGIC y registra cada una como tabla Delta en el esquema `workspace.fen_risk` para poder
# MAGIC consultarlas con Spark SQL en el notebook de análisis exploratorio.
# MAGIC
# MAGIC Fuentes:
# MAGIC 1. **NOAA** — Niño 1+2 y ONI (índice oficial CPC, reemplaza a Niño 3.4 que PSL descontinuó en 1994)
# MAGIC 2. **INDECI/SINPAD** — Emergencias y daños a nivel nacional (Plataforma Nacional de Datos Abiertos)
# MAGIC 3. **Meteorología** — 10 distritos de la provincia de Piura, 1997-2026 (Open-Meteo; SENAMHI no
# MAGIC    tiene estaciones en Piura dentro del dataset de datos abiertos disponible — ver nota más abajo)

# COMMAND ----------

import os
import time

import pandas as pd
import requests

VOLUME_PATH = "/Volumes/workspace/fen_risk/landing"
SCHEMA = "workspace.fen_risk"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")


def download(url, dest_path, headers=None):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    resp = requests.get(url, headers=headers or {}, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    print(f"OK  {dest_path}  ({len(resp.content) / 1024:.1f} KB)")
    return dest_path

# COMMAND ----------

# MAGIC %md ## 1. NOAA — índices oceánicos El Niño

# COMMAND ----------

download(
    "https://psl.noaa.gov/data/correlation/nina1.anom.csv",
    f"{VOLUME_PATH}/noaa/nina1_anom.csv",
)
# nina34.anom.csv de PSL está descontinuado desde 1994 (confirmado en exploración previa).
# Se usa el ONI (Oceanic Nino Index) de NOAA/CPC como equivalente oficial y vigente.
download(
    "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    f"{VOLUME_PATH}/noaa/oni_cpc.txt",
)

# COMMAND ----------

# MAGIC %md ## 2. INDECI / SINPAD — Emergencias y daños a nivel nacional
# MAGIC El portal de datos abiertos del Perú bloquea clientes sin User-Agent de navegador (WAF),
# MAGIC por eso se manda uno explícito.

# COMMAND ----------

download(
    "https://www.datosabiertos.gob.pe/sites/default/files/BD_2003-2025_EMERGENCIAS.csv",
    f"{VOLUME_PATH}/indeci/BD_2003-2025_EMERGENCIAS.csv",
    headers={"User-Agent": BROWSER_UA},
)

# COMMAND ----------

# MAGIC %md ## 3. Meteorología — 10 distritos de la provincia de Piura (Open-Meteo, 1997-2026)
# MAGIC
# MAGIC **Nota metodológica**: el dataset de SENAMHI publicado en datosabiertos.gob.pe
# MAGIC ("Variables Meteorológicas de las Estaciones automáticas de intercambio internacional")
# MAGIC solo cubre 5 estaciones en Arequipa/Lima/Tacna — cero cobertura en Piura. El portal oficial
# MAGIC de descarga interactiva de SENAMHI está en mantenimiento. Por eso se usa Open-Meteo
# MAGIC (reanálisis histórico, validado contra el evento conocido de lluvias del 26-mar-2017 en Piura:
# MAGIC 130.6mm ese día, coincide con el desborde histórico del río Piura).
# MAGIC
# MAGIC 7 distritos usan coordenada del centro urbano (geocoding); 3 (Cura Mori, El Tallán,
# MAGIC Veintiséis de Octubre) usan el centroide de su polígono del shapefile IGN, porque no
# MAGIC tienen entrada propia en la base de geocoding. Pendiente homogenizar a futuro con
# MAGIC extracción por polígono completo en vez de un punto único.

# COMMAND ----------

PIURA_DISTRITOS = {
    "piura": (-5.18192, -80.65715),
    "castilla": (-5.22658, -80.61579),
    "catacaos": (-5.26667, -80.68333),
    "la_arena": (-5.34659, -80.71078),
    "la_union": (-5.40232, -80.74224),
    "las_lomas": (-4.65333, -80.24667),
    "tambo_grande": (-4.92694, -80.34472),
    "cura_mori": (-5.357591, -80.586049),
    "el_tallan": (-5.441295, -80.611371),
    "veintiseis_octubre": (-5.161385, -80.696543),
}

for distrito, (lat, lon) in PIURA_DISTRITOS.items():
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&start_date=1997-01-01&end_date=2026-09-19"
        "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean"
        "&timezone=America%2FLima&format=csv"
    )
    dest = f"{VOLUME_PATH}/meteo_reanalysis/piura_{distrito}_1997_2026.csv"
    for attempt in range(5):
        try:
            download(url, dest)
            break
        except requests.HTTPError as e:
            print(f"  reintento {attempt + 1} para {distrito}: {e}")
            time.sleep(15)
    time.sleep(2)

# COMMAND ----------

# MAGIC %md ## 4. Registrar como tablas Delta en `workspace.fen_risk`

# COMMAND ----------

# --- NOAA Nino 1+2 ---
df_nina1 = (
    spark.read.option("header", True).option("inferSchema", True)
    .csv(f"{VOLUME_PATH}/noaa/nina1_anom.csv")
)
old_col = [c for c in df_nina1.columns if c != "Date"][0]
df_nina1 = df_nina1.withColumnRenamed("Date", "fecha").withColumnRenamed(old_col, "nina12_anom")
df_nina1.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.landing_noaa_nina12")

# --- NOAA ONI (formato de ancho fijo con espacios variables -> se parsea con pandas) ---
pdf_oni = pd.read_csv(f"{VOLUME_PATH}/noaa/oni_cpc.txt", sep=r"\s+")
pdf_oni.columns = ["temporada", "anio", "sst_total", "oni_anom"]
spark.createDataFrame(pdf_oni).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.landing_noaa_oni")

# --- INDECI / SINPAD (separador ; , encoding Latin-1) ---
df_indeci = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .option("delimiter", ";")
    .option("encoding", "ISO-8859-1")
    .csv(f"{VOLUME_PATH}/indeci/BD_2003-2025_EMERGENCIAS.csv")
)

# Delta no admite espacios ni caracteres especiales en nombres de columna;
# el CSV original de INDECI trae headers como "FECHA  DE LA EMER" o
# "CODIGO DE EMERGENCIA-SINPAD". Se normalizan antes de guardar.
import re


_ACCENTS = str.maketrans("ÁÉÍÓÚÑáéíóúñ", "AEIOUNaeioun")


def sanitize_col(c):
    c = c.strip().translate(_ACCENTS)
    c = re.sub(r"[^0-9a-zA-Z]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c.upper()


for c in df_indeci.columns:
    df_indeci = df_indeci.withColumnRenamed(c, sanitize_col(c))

df_indeci.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.landing_indeci_emergencias")

# --- Meteorología Piura (cada CSV trae 3 líneas de metadata de coordenadas antes del header real) ---
pdfs = []
for distrito in PIURA_DISTRITOS:
    path = f"{VOLUME_PATH}/meteo_reanalysis/piura_{distrito}_1997_2026.csv"
    pdf = pd.read_csv(path, skiprows=3)
    pdf.columns = ["fecha", "precip_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]
    pdf["distrito"] = distrito
    pdfs.append(pdf)
pdf_meteo = pd.concat(pdfs, ignore_index=True)
spark.createDataFrame(pdf_meteo).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.landing_meteo_piura")

print("Tablas Landing registradas en", SCHEMA)
for t in ["landing_noaa_nina12", "landing_noaa_oni", "landing_indeci_emergencias", "landing_meteo_piura"]:
    n = spark.table(f"{SCHEMA}.{t}").count()
    print(f"  {t}: {n:,} filas")
