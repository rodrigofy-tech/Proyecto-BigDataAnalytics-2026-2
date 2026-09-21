# Databricks notebook source
# MAGIC %md
# MAGIC # FEN-RISK Perú — 02. Análisis exploratorio (EDA) sobre Landing
# MAGIC
# MAGIC Antes de pasar a la capa Curated (limpieza, tipado, JOIN), este notebook responde
# MAGIC lo que la propuesta original define como evidencia mínima de la primera entrega:
# MAGIC número de registros, esquema, nulos, duplicados y una primera mirada a las
# MAGIC distribuciones clave de cada fuente. Corre sobre las tablas Delta creadas en
# MAGIC `01_landing_ingesta`.

# COMMAND ----------

SCHEMA = "workspace.fen_risk"

# COMMAND ----------

# MAGIC %md ## 1. NOAA — índices oceánicos El Niño

# COMMAND ----------

df_nina12 = spark.table(f"{SCHEMA}.landing_noaa_nina12")
df_oni = spark.table(f"{SCHEMA}.landing_noaa_oni")

print(f"Niño 1+2: {df_nina12.count():,} filas")
df_nina12.printSchema()
df_nina12.describe("nina12_anom").show()

print(f"ONI: {df_oni.count():,} filas")
df_oni.printSchema()
df_oni.describe("oni_anom").show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Calidad de datos**: Niño 1+2 usa `-99.99` como marcador de "sin dato" (meses aún no
# MAGIC observados / futuro). Se cuentan para dimensionar cuántos registros habrá que filtrar en Curated.

# COMMAND ----------

from pyspark.sql import functions as F

n_missing_nina12 = df_nina12.filter(F.col("nina12_anom") < -90).count()
print(f"Registros 'sin dato' en Niño 1+2: {n_missing_nina12:,} de {df_nina12.count():,}")

display(
    df_oni.filter(F.col("anio") >= 2015)
    .orderBy("anio")
    .select("temporada", "anio", "oni_anom")
)

# COMMAND ----------

# MAGIC %md ## 2. INDECI / SINPAD — Emergencias y daños

# COMMAND ----------

df_indeci = spark.table(f"{SCHEMA}.landing_indeci_emergencias")

print(f"Total nacional: {df_indeci.count():,} filas")
print(f"Columnas: {len(df_indeci.columns)}")
df_indeci.printSchema()

# COMMAND ----------

# MAGIC %md ### Nulos por columna clave

# COMMAND ----------

cols_clave = ["FECHA_DE_LA_EMER", "DPTO", "PROV", "DIST", "PELIGRO", "FALLECIDOS", "AFECTADOS"]
df_indeci.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols_clave]
).show(truncate=False)

# COMMAND ----------

# MAGIC %md ### Duplicados
# MAGIC Se usa el código de emergencia SINPAD como llave natural.

# COMMAND ----------

total = df_indeci.count()
distintos = df_indeci.select("CODIGO_DE_EMERGENCIA_SINPAD").distinct().count()
print(f"Filas totales: {total:,} | códigos SINPAD distintos: {distintos:,} | duplicados: {total - distintos:,}")

# COMMAND ----------

# MAGIC %md ### Piura: emergencias por año y por tipo de peligro

# COMMAND ----------

df_piura = df_indeci.filter(F.col("DPTO") == "PIURA")
print(f"Emergencias registradas en Piura (2003-2025): {df_piura.count():,}")

display(
    df_piura.groupBy("ANO").count().orderBy("ANO")
)

display(
    df_piura.groupBy("PELIGRO").count().orderBy(F.desc("count")).limit(10)
)

display(
    df_piura.groupBy("DIST").agg(
        F.count("*").alias("n_emergencias"),
        F.sum("AFECTADOS").alias("total_afectados"),
        F.sum("DAMNIFICADOS").alias("total_damnificados"),
    ).orderBy(F.desc("n_emergencias")).limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Lectura rápida**: si 2017 y 2023 destacan como picos de emergencias en Piura, confirma
# MAGIC que el dataset captura bien los episodios de El Niño Costero conocidos — dato relevante
# MAGIC para justificar el dataset ante el profesor.

# COMMAND ----------

# MAGIC %md ## 3. Meteorología — 10 distritos de Piura (1997-2026)

# COMMAND ----------

df_meteo = spark.table(f"{SCHEMA}.landing_meteo_piura")

print(f"Total filas: {df_meteo.count():,} ({df_meteo.select('distrito').distinct().count()} distritos)")
df_meteo.printSchema()

# COMMAND ----------

# MAGIC %md ### Nulos y duplicados

# COMMAND ----------

df_meteo.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df_meteo.columns]
).show()

total_meteo = df_meteo.count()
distintos_meteo = df_meteo.select("distrito", "fecha").distinct().count()
print(f"Filas: {total_meteo:,} | combinaciones (distrito, fecha) distintas: {distintos_meteo:,} | duplicados: {total_meteo - distintos_meteo:,}")

# COMMAND ----------

# MAGIC %md ### Distribución de precipitación por distrito

# COMMAND ----------

display(
    df_meteo.groupBy("distrito").agg(
        F.round(F.avg("precip_mm"), 2).alias("precip_prom_mm"),
        F.round(F.max("precip_mm"), 2).alias("precip_max_mm"),
        F.round(F.avg("temp_max_c"), 1).alias("temp_max_prom_c"),
        F.round(F.avg("humedad_rel_pct"), 1).alias("humedad_prom_pct"),
    ).orderBy(F.desc("precip_max_mm"))
)

# COMMAND ----------

# MAGIC %md ### Validación contra un evento histórico conocido
# MAGIC El 26 de marzo de 2017 el río Piura se desbordó en un episodio muy documentado del
# MAGIC Niño Costero 2017. Si el dato de precipitación de ese día es consistente con un pico
# MAGIC extremo, valida que la fuente (Open-Meteo) captura bien la señal real.

# COMMAND ----------

display(
    df_meteo.filter((F.col("fecha") >= "2017-03-20") & (F.col("fecha") <= "2017-03-31"))
    .filter(F.col("distrito") == "piura")
    .orderBy("fecha")
    .select("fecha", "precip_mm", "temp_max_c", "humedad_rel_pct")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen para la entrega
# MAGIC
# MAGIC | Resultado | Qué demuestra |
# MAGIC |---|---|
# MAGIC | Conteo de filas por fuente | Ingesta / Volumen |
# MAGIC | Esquema (`printSchema`) | Estructura original |
# MAGIC | Nulos por columna clave | Data Quality |
# MAGIC | Duplicados por llave natural | Deduplicación necesaria en Curated |
# MAGIC | Emergencias Piura por año/tipo/distrito | Relevancia del dataset para el caso de uso |
# MAGIC | Precipitación por distrito + validación evento 2017 | Confiabilidad de la fuente meteorológica |
# MAGIC
# MAGIC Con esto queda documentado qué hay que limpiar (nulos, duplicados, tipos) antes de
# MAGIC construir la capa **Curated** en el siguiente notebook.
