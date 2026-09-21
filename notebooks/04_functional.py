# Databricks notebook source
# MAGIC %md
# MAGIC # FEN-RISK Perú — 04. Functional: JOIN, window functions e indicador de riesgo
# MAGIC
# MAGIC Toma las tablas `curated_*` y arma la tabla final `fen_risk_functional`:
# MAGIC una fila por (distrito, fecha) con clima + señal El Niño + evidencia histórica
# MAGIC de emergencias, lista para análisis o para entrenar un modelo más adelante.
# MAGIC
# MAGIC Pasos:
# MAGIC 1. Cargar tablas Curated
# MAGIC 2. Cruzar meteorología con Niño 1+2 (por año-mes)
# MAGIC 3. Cruzar con ONI (por temporada trimestral — más delicado, se explica abajo)
# MAGIC 4. Calcular acumulados de precipitación con window functions (1d/3d/7d)
# MAGIC 5. Calcular promedio histórico mensual de precipitación por distrito
# MAGIC 6. Agregar emergencias INDECI por (distrito, día) y cruzar
# MAGIC 7. Construir el indicador de riesgo
# MAGIC 8. Guardar la tabla Functional final

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

SCHEMA = "workspace.fen_risk"

# COMMAND ----------

# MAGIC %md ## Paso 1: Cargar tablas Curated

# COMMAND ----------

df_meteo = spark.table(f"{SCHEMA}.curated_meteo_piura")
df_nina12 = spark.table(f"{SCHEMA}.curated_noaa_nina12")
df_oni = spark.table(f"{SCHEMA}.curated_noaa_oni")
df_indeci = spark.table(f"{SCHEMA}.curated_indeci_piura")

print(f"meteo: {df_meteo.count():,} | nina12: {df_nina12.count():,} | oni: {df_oni.count():,} | indeci: {df_indeci.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Cruzar meteorología con Niño 1+2
# MAGIC Niño 1+2 es mensual; se cruza por (año, mes) del día de meteorología.

# COMMAND ----------

df_meteo = df_meteo.withColumn("anio", F.year("fecha")).withColumn("mes", F.month("fecha"))

nina12_mensual = df_nina12.withColumn("anio", F.year("fecha")).withColumn("mes", F.month("fecha")).select(
    "anio", "mes", F.col("nina12_anom")
)

df_meteo = df_meteo.join(nina12_mensual, on=["anio", "mes"], how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Cruzar con ONI (temporada trimestral)
# MAGIC El ONI no es mensual sino por temporada móvil de 3 meses (DJF, JFM, FMA...).
# MAGIC Se usa la temporada cuyo mes central coincide con el mes del registro. El único
# MAGIC caso especial es diciembre: pertenece a la temporada NDJ, pero esa fila en la
# MAGIC tabla del NOAA está etiquetada con el año del enero siguiente (ej. diciembre de
# MAGIC 2016 cae en "NDJ 2017"), así que ese mes hay que sumarle 1 al año antes de cruzar.

# COMMAND ----------

MES_A_TEMPORADA = {
    1: "DJF", 2: "JFM", 3: "FMA", 4: "MAM", 5: "AMJ", 6: "MJJ",
    7: "JJA", 8: "JAS", 9: "ASO", 10: "SON", 11: "OND", 12: "NDJ",
}

mapping_temporada = F.create_map([F.lit(x) for pair in MES_A_TEMPORADA.items() for x in pair])

df_meteo = df_meteo.withColumn("temporada_oni", mapping_temporada[F.col("mes")])
df_meteo = df_meteo.withColumn(
    "anio_oni", F.when(F.col("mes") == 12, F.col("anio") + 1).otherwise(F.col("anio"))
)

oni_lookup = df_oni.select(
    F.col("temporada").alias("temporada_oni"),
    F.col("anio").alias("anio_oni"),
    "oni_anom",
)

df_meteo = df_meteo.join(oni_lookup, on=["temporada_oni", "anio_oni"], how="left")

n_sin_oni = df_meteo.filter(F.col("oni_anom").isNull()).count()
print(f"Filas sin match de ONI (esperable en los extremos de la serie): {n_sin_oni:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Acumulados de precipitación (window functions)
# MAGIC `precip_1d` ya es la columna original; se agregan 3 y 7 días acumulados por
# MAGIC distrito, ordenados por fecha — igual que se planteó en la propuesta original.

# COMMAND ----------

w_distrito_fecha = Window.partitionBy("distrito").orderBy("fecha")

df_meteo = df_meteo.withColumn("precip_1d", F.col("precip_mm"))
df_meteo = df_meteo.withColumn(
    "precip_3d", F.sum("precip_mm").over(w_distrito_fecha.rowsBetween(-2, 0))
)
df_meteo = df_meteo.withColumn(
    "precip_7d", F.sum("precip_mm").over(w_distrito_fecha.rowsBetween(-6, 0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5: Promedio histórico mensual de precipitación por distrito
# MAGIC Sirve para comparar "cuánto llovió este día" contra "cuánto llueve normalmente
# MAGIC ese mes en ese distrito" — la base de un indicador de anomalía.

# COMMAND ----------

w_distrito_mes = Window.partitionBy("distrito", "mes")

df_meteo = df_meteo.withColumn(
    "precip_promedio_historico_mes", F.avg("precip_mm").over(w_distrito_mes)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6: Emergencias INDECI por (distrito, día) y cruce
# MAGIC INDECI puede tener más de un registro el mismo día en el mismo distrito
# MAGIC (distintos tipos de peligro); se agregan antes de cruzar para que la tabla final
# MAGIC siga teniendo una sola fila por (distrito, fecha).
# MAGIC
# MAGIC **Hallazgo del EDA**: INDECI mezcla tipos de peligro sin relación con El Niño
# MAGIC Costero (sequía, incendios, sismos...). `SEQUIA` incluso es la señal opuesta —
# MAGIC son días sin lluvia. Por eso se agregan dos versiones: una general (`n_emergencias`,
# MAGIC cualquier peligro) y otra específica (`n_emergencias_lluvia`) solo con los tipos
# MAGIC realmente asociados a precipitación/El Niño Costero: `LLUVIA INTENSA`,
# MAGIC `INUNDACION`, `DESLIZAMIENTO`, `HUAYCO`. Esta segunda es la que debería usarse
# MAGIC como variable objetivo de un futuro modelo.

# COMMAND ----------

PELIGROS_LLUVIA = ["LLUVIA INTENSA", "INUNDACION", "DESLIZAMIENTO", "HUAYCO"]

indeci_diario = df_indeci.groupBy(
    F.col("ubigeo").cast("string").alias("ubigeo"),
    F.col("fecha_emergencia").alias("fecha"),
).agg(
    F.count("*").alias("n_emergencias"),
    F.sum("FALLECIDOS").alias("fallecidos"),
    F.sum("AFECTADOS").alias("afectados"),
    F.sum("DAMNIFICADOS").alias("damnificados"),
    F.sum(F.col("viviendas_destruidas") + F.col("viviendas_afectadas")).alias("viviendas_impactadas"),
    F.count(F.when(F.col("PELIGRO").isin(PELIGROS_LLUVIA), 1)).alias("n_emergencias_lluvia"),
    F.sum(F.when(F.col("PELIGRO").isin(PELIGROS_LLUVIA), F.col("AFECTADOS")).otherwise(0)).alias("afectados_lluvia"),
    F.sum(F.when(F.col("PELIGRO").isin(PELIGROS_LLUVIA), F.col("DAMNIFICADOS")).otherwise(0)).alias("damnificados_lluvia"),
)

df_functional = df_meteo.join(indeci_diario, on=["ubigeo", "fecha"], how="left")

# los distritos-día sin emergencia no aparecen en INDECI -> null; se convierten en 0
cols_indeci = [
    "n_emergencias", "fallecidos", "afectados", "damnificados", "viviendas_impactadas",
    "n_emergencias_lluvia", "afectados_lluvia", "damnificados_lluvia",
]
for c in cols_indeci:
    df_functional = df_functional.withColumn(c, F.coalesce(F.col(c), F.lit(0)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7: Indicador de riesgo
# MAGIC Regla simple e inicial (documentada como punto de partida, no como verdad
# MAGIC definitiva — se puede refinar cuando entren las imágenes satelitales). Se calculan
# MAGIC dos versiones: general (`emergencia`, `nivel_impacto`, cualquier tipo de peligro,
# MAGIC útil como contexto) y específica de lluvia (`emergencia_lluvia`,
# MAGIC `nivel_impacto_lluvia`, la que debería usarse para responder la pregunta de
# MAGIC investigación del proyecto).
# MAGIC - `emergencia*` = 1 si hubo al menos una emergencia ese día en ese distrito.
# MAGIC - `nivel_impacto*`: 0 sin afectación, 1 baja (≤50 personas), 2 media (≤500), 3 alta (>500).

# COMMAND ----------

df_functional = df_functional.withColumn(
    "emergencia", F.when(F.col("n_emergencias") > 0, 1).otherwise(0)
).withColumn(
    "emergencia_lluvia", F.when(F.col("n_emergencias_lluvia") > 0, 1).otherwise(0)
)

df_functional = df_functional.withColumn(
    "personas_impactadas", F.col("afectados") + F.col("damnificados")
).withColumn(
    "personas_impactadas_lluvia", F.col("afectados_lluvia") + F.col("damnificados_lluvia")
)


def nivel_impacto(col):
    return (
        F.when(F.col(col) == 0, 0)
        .when(F.col(col) <= 50, 1)
        .when(F.col(col) <= 500, 2)
        .otherwise(3)
    )


df_functional = df_functional.withColumn(
    "nivel_impacto", nivel_impacto("personas_impactadas")
).withColumn(
    "nivel_impacto_lluvia", nivel_impacto("personas_impactadas_lluvia")
)

# COMMAND ----------

# MAGIC %md ## Paso 8: Guardar la tabla Functional final

# COMMAND ----------

df_functional_final = df_functional.select(
    "fecha", "ubigeo", "distrito", "distrito_nombre",
    "precip_1d", "precip_3d", "precip_7d", "precip_promedio_historico_mes",
    "temp_max_c", "temp_min_c", "humedad_rel_pct",
    "nina12_anom", "oni_anom",
    "n_emergencias", "fallecidos", "afectados", "damnificados", "viviendas_impactadas",
    "emergencia", "personas_impactadas", "nivel_impacto",
    "n_emergencias_lluvia", "afectados_lluvia", "damnificados_lluvia",
    "emergencia_lluvia", "personas_impactadas_lluvia", "nivel_impacto_lluvia",
)

(df_functional_final
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{SCHEMA}.fen_risk_functional"))

n = spark.table(f"{SCHEMA}.fen_risk_functional").count()
n_lluvia = spark.table(f"{SCHEMA}.fen_risk_functional").filter(F.col("emergencia_lluvia") == 1).count()
n_general = spark.table(f"{SCHEMA}.fen_risk_functional").filter(F.col("emergencia") == 1).count()
print(f"fen_risk_functional: {n:,} filas")
print(f"  con emergencia (cualquier tipo): {n_general:,}")
print(f"  con emergencia_lluvia (solo El Niño Costero): {n_lluvia:,}")

display(
    spark.table(f"{SCHEMA}.fen_risk_functional")
    .filter(F.col("emergencia_lluvia") == 1)
    .orderBy(F.desc("personas_impactadas_lluvia"))
    .limit(10)
)
