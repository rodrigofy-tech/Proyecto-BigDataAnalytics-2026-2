# Databricks notebook source
# MAGIC %md
# MAGIC # FEN-RISK Perú — 03. Curated: limpieza, tipado y llave de cruce
# MAGIC
# MAGIC Toma las tablas `landing_*` (crudas, tal cual llegaron) y las deja listas para el
# MAGIC JOIN de la capa Functional. No se agrega nada todavía (eso es Functional) — acá
# MAGIC solo se limpia, tipa, deduplica y se arma la llave común (UBIGEO) entre fuentes.
# MAGIC
# MAGIC Pasos:
# MAGIC 1. Cargar tablas Landing
# MAGIC 2. Corregir tipos de datos (fechas, numéricos)
# MAGIC 3. Filtrar valores "sin dato" (sentinels)
# MAGIC 4. Deduplicar por llave natural
# MAGIC 5. Agregar UBIGEO a la tabla de meteorología (no lo traía)
# MAGIC 6. Acotar INDECI al alcance del proyecto (Piura)
# MAGIC 7. Quedarnos solo con las columnas relevantes
# MAGIC 8. Guardar como tablas Curated (Delta)

# COMMAND ----------

from pyspark.sql import functions as F

SCHEMA = "workspace.fen_risk"

# COMMAND ----------

# MAGIC %md ## Paso 1: Cargar tablas Landing

# COMMAND ----------

df_nina12 = spark.table(f"{SCHEMA}.landing_noaa_nina12")
df_oni = spark.table(f"{SCHEMA}.landing_noaa_oni")
df_indeci = spark.table(f"{SCHEMA}.landing_indeci_emergencias")
df_meteo = spark.table(f"{SCHEMA}.landing_meteo_piura")

print(f"nina12: {df_nina12.count():,} | oni: {df_oni.count():,} | indeci: {df_indeci.count():,} | meteo: {df_meteo.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Corregir tipos de datos
# MAGIC `FECHA_DE_LA_EMER` (INDECI) y `fecha` (meteo, nina12) llegaron como texto —
# MAGIC se convierten a `DATE` real para poder filtrar/ordenar/hacer JOIN por fecha.
# MAGIC
# MAGIC **Hallazgo**: la fecha de INDECI no tiene un formato único — hay filas en
# MAGIC `dd/MM/yyyy` (ej. `11/02/2003`) y otras en `MM/dd/yy` (ej. `02/15/25`), seguramente
# MAGIC porque el dataset se fue exportando en distintos lotes a lo largo de los años.
# MAGIC Se usa `try_to_date` con varios formatos candidatos (en vez de `to_date` con uno
# MAGIC solo, que revienta con `ANSI` activado) y se toma el primero que calce.

# COMMAND ----------

FORMATOS_FECHA = ["dd/MM/yyyy", "MM/dd/yy", "dd/MM/yy", "MM/dd/yyyy"]

df_indeci = df_indeci.withColumn(
    "fecha_emergencia",
    F.coalesce(*[
        F.expr(f"try_to_date(FECHA_DE_LA_EMER, '{fmt}')") for fmt in FORMATOS_FECHA
    ]),
)

df_meteo = df_meteo.withColumn("fecha", F.to_date("fecha", "yyyy-MM-dd"))

df_nina12 = df_nina12.withColumn("fecha", F.to_date("fecha", "yyyy-MM-dd"))

# sanity check: cuántas fechas de INDECI no calzaron con ninguno de los formatos probados
n_fechas_invalidas = df_indeci.filter(
    F.col("FECHA_DE_LA_EMER").isNotNull() & F.col("fecha_emergencia").isNull()
).count()
print(f"Fechas INDECI que no pudieron convertirse con ningún formato: {n_fechas_invalidas:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Filtrar valores "sin dato" (sentinels)
# MAGIC Niño 1+2 usa `-99.99` (y en la práctica vimos `-9999`) para marcar meses sin
# MAGIC observación todavía. Se convierten a `NULL` real en vez de dejarlos como número.

# COMMAND ----------

df_nina12 = df_nina12.withColumn(
    "nina12_anom",
    F.when(F.col("nina12_anom") < -90, None).otherwise(F.col("nina12_anom")),
)

n_nulos_nina12 = df_nina12.filter(F.col("nina12_anom").isNull()).count()
print(f"Meses sin dato en Niño 1+2 (convertidos a NULL): {n_nulos_nina12:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Deduplicar por llave natural
# MAGIC INDECI por código SINPAD (una emergencia = un código único).
# MAGIC Meteorología por (distrito, fecha) — no debería haber más de un registro diario.

# COMMAND ----------

antes = df_indeci.count()
df_indeci = df_indeci.dropDuplicates(["CODIGO_DE_EMERGENCIA_SINPAD"])
print(f"INDECI: {antes:,} -> {df_indeci.count():,} filas (duplicados eliminados: {antes - df_indeci.count():,})")

antes = df_meteo.count()
df_meteo = df_meteo.dropDuplicates(["distrito", "fecha"])
print(f"Meteo: {antes:,} -> {df_meteo.count():,} filas (duplicados eliminados: {antes - df_meteo.count():,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5: Agregar UBIGEO a la tabla de meteorología
# MAGIC La tabla de meteorología solo trae el nombre del distrito (`piura`, `catacaos`...);
# MAGIC INDECI cruza por `COD_DISTRITO` (UBIGEO). Sin este paso, el JOIN de la capa
# MAGIC Functional no podría cruzar ambas fuentes correctamente. El UBIGEO viene del
# MAGIC shapefile oficial del IGN (mismo que usamos para los centroides de los 3
# MAGIC distritos sin geocoding).

# COMMAND ----------

UBIGEO_PIURA = {
    "piura": ("200101", "PIURA"),
    "castilla": ("200104", "CASTILLA"),
    "catacaos": ("200105", "CATACAOS"),
    "cura_mori": ("200107", "CURA MORI"),
    "el_tallan": ("200108", "EL TALLAN"),
    "la_arena": ("200109", "LA ARENA"),
    "la_union": ("200110", "LA UNION"),
    "las_lomas": ("200111", "LAS LOMAS"),
    "tambo_grande": ("200114", "TAMBO GRANDE"),
    "veintiseis_octubre": ("200115", "VEINTISEIS DE OCTUBRE"),
}

mapping_rows = [(k, v[0], v[1]) for k, v in UBIGEO_PIURA.items()]
df_ubigeo = spark.createDataFrame(mapping_rows, ["distrito", "ubigeo", "distrito_nombre"])

df_meteo = df_meteo.join(df_ubigeo, on="distrito", how="left")

n_sin_ubigeo = df_meteo.filter(F.col("ubigeo").isNull()).count()
print(f"Filas de meteo sin UBIGEO asignado (debería ser 0): {n_sin_ubigeo:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6: Acotar INDECI al alcance del proyecto (Piura)
# MAGIC INDECI es nacional (142K filas); el proyecto trabaja Piura. Se filtra y se
# MAGIC materializa aparte para no repetir el filtro en cada consulta downstream.

# COMMAND ----------

df_indeci_piura = df_indeci.filter(F.col("DPTO") == "PIURA")
print(f"INDECI Piura: {df_indeci_piura.count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7: Quedarnos solo con las columnas relevantes
# MAGIC INDECI trae ~48 columnas (ganado, canales de regadío, etc.) que no aportan a un
# MAGIC indicador de riesgo por lluvias. Se seleccionan las que sí se van a usar.

# COMMAND ----------

df_indeci_piura = df_indeci_piura.select(
    F.col("CODIGO_DE_EMERGENCIA_SINPAD").alias("codigo_emergencia"),
    "fecha_emergencia",
    F.col("COD_DISTRITO").alias("ubigeo"),
    F.col("DPTO").alias("departamento"),
    F.col("PROV").alias("provincia"),
    F.col("DIST").alias("distrito_nombre"),
    F.col("PELIGRO").alias("peligro"),
    F.col("TIPO_DE_PELIGRO").alias("tipo_peligro"),
    "FALLECIDOS",
    "DESAPARECIDOS",
    "LESIONADOS",
    "DAMNIFICADOS",
    "AFECTADOS",
    F.col("VIVIENDAS_DESTRUIDAS").alias("viviendas_destruidas"),
    F.col("VIVIENDAS_AFECTADAS").alias("viviendas_afectadas"),
)

df_meteo = df_meteo.select(
    "ubigeo",
    "distrito",
    "distrito_nombre",
    "fecha",
    "precip_mm",
    "temp_max_c",
    "temp_min_c",
    "humedad_rel_pct",
)

# COMMAND ----------

# MAGIC %md ## Paso 8: Guardar como tablas Curated (Delta)

# COMMAND ----------

(df_nina12.select("fecha", "nina12_anom")
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{SCHEMA}.curated_noaa_nina12"))

(df_oni
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{SCHEMA}.curated_noaa_oni"))

(df_indeci_piura
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{SCHEMA}.curated_indeci_piura"))

(df_meteo
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{SCHEMA}.curated_meteo_piura"))

print("Tablas Curated listas en", SCHEMA)
for t in ["curated_noaa_nina12", "curated_noaa_oni", "curated_indeci_piura", "curated_meteo_piura"]:
    n = spark.table(f"{SCHEMA}.{t}").count()
    print(f"  {t}: {n:,} filas")
