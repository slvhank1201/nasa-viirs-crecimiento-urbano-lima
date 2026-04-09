# NASA VIIRS: Análisis y Predicción de Expansión Urbana en Lima Metropolitana 

Un estudio de ciencia de datos geoespaciales que procesa telemetría satelital cruda (misión Suomi NPP / VIIRS de la NASA) para evaluar, cuantificar y predecir la densificación y el crecimiento urbano de la capital peruana hasta el año 2030.

## Síntesis del Proyecto
El crecimiento de las megalópolis latinoamericanas suele medirse con censos desfasados. Este proyecto propone un enfoque empírico y automatizado: utilizar las emisiones de luz nocturna (Radiancia) como proxy directo del desarrollo urbano y la actividad económica. 

Mediante el procesamiento de matrices de datos HDF5 en Python y modelos de regresión, se ha logrado mapear la evolución de Lima (2019-2023) y proyectar su huella lumínica, consolidando los hallazgos en un dashboard interactivo gerencial.

## Hallazgos Clave y Análisis de Tendencias

El análisis de la huella lumínica revela patrones críticos sobre el desarrollo territorial de Lima:

* **Hiperconcentración del Eje Financiero (Saturación del Núcleo):** La interpretación visual y cuantitativa de los mapas de calor revela una aglomeración crítica de radiancia máxima en la franja costera-central (eje San Isidro, Miraflores, Lima Centro). Este clúster representa el núcleo corporativo y comercial consolidado de la capital.
* **Dinámica de Expansión Periférica (El Delta):** Al cruzar el mapa del núcleo consolidado con el mapa de Delta Radiancia (2019-2023), se evidencia que, si bien el eje central mantiene la mayor intensidad absoluta, la tasa de crecimiento de nueva infraestructura se está desplazando hacia las periferias y zonas peri-urbanas. Esto marca claramente los nuevos vectores de demanda para planificación de servicios públicos y logística estatal.
* **Densificación Comercial acelerada:** La proyección muestra que los píxeles clasificados como *Centro Comercial* (radiancia > 120 nW/cm²/sr) superarán en volumen a los de zonas *Rurales*. Esto indica que Lima no solo se expande horizontalmente, sino que sus periferias están mutando agresivamente hacia núcleos de alta intensidad económica y comercial.
* **Consolidación Territorial (Cobertura del 59.1%):** Más de la mitad del territorio evaluado ya supera el umbral de actividad urbana detectable (10 nW/cm²/sr). El suelo rural y peri-urbano está siendo absorbido a un ritmo de expansión acumulada del ~3% anual.
* **Fluctuaciones de Intensidad (Variación YoY):** Aunque la tendencia general de la Radiancia Media es alcista (mayor densidad), se registran variaciones interanuales negativas en periodos específicos. Esto refleja la sensibilidad del modelo para captar externalidades (desaceleraciones económicas) o el recambio de infraestructura pública (transición a luces LED).

## Arquitectura Técnica y Metodología

El pipeline de datos está construido para manejar telemetría satelital a escala:

1.  **Extracción (API Earthdata):** Consulta y descarga automatizada de productos `VNP46A3` (Black Marble) filtrados por Bounding Box (Lima).
2.  **Procesamiento y Limpieza (Python):** * Uso de `h5py` para lectura de tensores espaciales.
    * Limpieza de anomalías (cloud cover, ruido de sensores) usando `numpy` y `pandas`.
    * Interpolación espacial (`scipy.interpolate.griddata`) para homogeneizar la cuadrícula.
3.  **Modelamiento Predictivo:** Implementación de modelos de **Regresión Lineal** (`scikit-learn`) sobre la varianza histórica para estimar la luminosidad futura por píxel hasta 2030.
4.  **Visualización (Power BI):** Integración de scripts nativos de Python (`matplotlib`) dentro de Power BI para renderizar mapas de calor dinámicos acoplados a segmentadores temporales (DAX).

---
*Desarrollado como iniciativa de Investigación de Operaciones aplicada a la optimización de recursos y planificación territorial.*
