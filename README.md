# Proyecto-BigDataAnalytics-2026-2

Proyecto para el curso de Big Data Analytics en el ciclo 2026-2 para la Universidad del Pacífico (UP).

## FEN-RISK Perú

Plataforma Big Data para integrar información meteorológica, oceánica (El Niño) y registros
históricos de emergencias, con el fin de identificar patrones de riesgo territorial asociados
a impactos del Fenómeno El Niño Costero en la costa peruana. Piloto actual: **provincia de
Piura**. Ver `Propuesta Big Data El Niño.pdf` para el planteamiento completo del problema,
justificación y arquitectura conceptual.

## Estado actual

Pipeline funcionando end-to-end en Databricks, por capas (`Landing → Curated → Functional`),
implementado en `notebooks/`:

| Notebook | Qué hace |
|---|---|
| `01_landing_ingesta.py` | Descarga las fuentes crudas desde sus URLs originales y las deja en el Volume `workspace.fen_risk.landing`, registrándolas también como tablas Delta `landing_*` |
| `02_eda_exploratorio.py` | Análisis exploratorio breve: conteo de filas, esquema, nulos, duplicados, y una validación cruzada contra el evento histórico del 26-mar-2017 en Piura |
| `03_curated.py` | Limpieza, tipado, deduplicación y estandarización de la llave de cruce (UBIGEO) entre fuentes |
| `04_functional.py` | JOIN final + window functions (precipitación acumulada 3d/7d, promedio histórico mensual) + indicador de riesgo por distrito-día → tabla `fen_risk_functional` |

Todo vive en Databricks bajo el esquema `workspace.fen_risk` (tablas) y el volume
`workspace.fen_risk.landing` (archivos crudos). Los datos **no se versionan en git**
(ver `.gitignore`) porque los notebooks los vuelven a descargar en cada corrida.

### Fuentes de datos

| Fuente | Cobertura | Notas |
|---|---|---|
| NOAA — Niño 1+2 | Mensual, histórico completo | `nina1.anom.csv` |
| NOAA — ONI (CPC) | Trimestral móvil, histórico completo | Reemplaza al índice Niño 3.4 de PSL, que está descontinuado desde 1994 |
| INDECI / SINPAD | Nacional, filtrado a Piura (6,065 registros) | Mezcla tipos de peligro no relacionados a lluvia (sequía, incendios, sismos); se separó un indicador específico `emergencia_lluvia` (solo `LLUVIA INTENSA`, `INUNDACION`, `DESLIZAMIENTO`, `HUAYCO`) |
| Meteorología | 10 distritos de la provincia de Piura, diario 1997-2026 | SENAMHI no tiene cobertura de estaciones en Piura en el dataset de datos abiertos disponible ni el portal de descarga interactiva (en mantenimiento); se usa **Open-Meteo** (reanálisis histórico), validado contra el evento real del 26-mar-2017 |
| Shapefile distrital (IGN) | Perú completo | `DISTRITOS_LIMITES.zip`, usado para los centroides de 3 distritos sin geocoding directo y, más adelante, para recortar imágenes satelitales |

### Resultado clave (validación del pipeline)

El registro de mayor impacto en `fen_risk_functional` es **27-mar-2017, distrito de
Catacaos**: 252mm de precipitación acumulada en 7 días, anomalía Niño 1+2 de 1.73,
67,894 personas impactadas — coincide exactamente con el desborde histórico documentado
del río Piura durante el Niño Costero 2017.

## Acceso

- **GitHub**: este repositorio.
- **Databricks**: workspace `dbc-36b5ed0a-3eb5.cloud.databricks.com`. Notebooks en
  `/Workspace/Users/ja.rivasr@alum.up.edu.pe/fen_risk/notebooks/`, tablas en
  `workspace.fen_risk`. Acceso de edición otorgado a los correos del equipo.

## Próximos pasos

1. **Imágenes satelitales** (Sentinel-1/2) por distrito y fecha de evento — pipeline propio
   Landing (imagen cruda) → Curated (features tipo NDWI extraídos con el shapefile del IGN)
   → se suman a `fen_risk_functional` con un JOIN por `(ubigeo, fecha)`.
2. **Completar coordenadas** de Cura Mori, El Tallán y Veintiséis de Octubre con
   extracción por polígono completo (no solo el centroide), igual criterio que se usará
   para recortar las imágenes.
3. **Etapa 2 — Modelo**: clasificación binaria sobre `emergencia_lluvia` (no multiclase,
   por el desbalance de clases: ~305 días positivos de 108,540). Comparar Logistic
   Regression (interpretable) vs Random Forest (Spark MLlib) sobre las features de
   `fen_risk_functional`.
4. Evaluar ampliar el piloto de Piura al resto del departamento o a otros departamentos
   de la costa norte (Tumbes, Lambayeque, La Libertad), una vez validado el enfoque.
