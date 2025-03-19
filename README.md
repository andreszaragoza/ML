# Modelo de Machine Learning de Videojuegos


![Procesamiento de Datos](notebooks\IMG\DALL·E 2025-03-11 15.40.14 - A futuristic Helldiver equipped with Machine Learning technology, wearing high-tech armor displaying holographic interfaces and active artificial inte.webp)

## Descripción General
Este notebook realiza un análisis y procesamiento de datos para clasificar juegos en diferentes categorías de rating. Utiliza técnicas de preprocesamiento, creación de características derivadas, imputación de valores nulos y entrenamiento de modelos de clasificación. El objetivo principal es predecir la categoría de rating de un juego basado en sus características.

## Objetivos
- Preprocesar un conjunto de datos de juegos, incluyendo la imputación de valores nulos y la creación de nuevas características.
- Clasificar los juegos en tres categorías de rating: **Bajo, Medio y Alto**.
- Entrenar un modelo de clasificación utilizando **LightGBM** y optimizar sus hiperparámetros con **RandomizedSearchCV**.
- Evaluar el modelo en términos de métricas de clasificación como **accuracy, f1-score** y **matriz de confusión**.

## Pasos Principales

### Carga y Exploración de Datos
- Se cargan los datos desde un archivo CSV (**Juegorawg_limpio.csv**).
- Se analizan las dimensiones del dataset y las estadísticas descriptivas de la columna **rating**.

### Preprocesamiento
- Se crean categorías de rating (**Bajo, Medio, Alto**) utilizando la función `pd.cut`.
- Se imputan valores nulos en la columna **rating_category** basándose en los valores de **rating**.

### Creación de Características Derivadas
Se generan nuevas columnas como:
- **rating_to_count_ratio**: Relación entre el rating y el número de calificaciones.
- **reviews_to_ratings_ratio**: Relación entre reseñas y calificaciones.
- **popularity_score**: Puntuación de popularidad basada en varias métricas.
- **recency_factor**: Factor de recencia basado en el año de lanzamiento.

### Selección y Codificación de Características
- Se seleccionan características numéricas y categóricas relevantes.
- Se codifican las características categóricas utilizando `LabelEncoder`.

### Entrenamiento del Modelo
- Se utiliza **LightGBM** como modelo base.
- Se optimizan los hiperparámetros con **RandomizedSearchCV**.
- Los mejores parámetros encontrados incluyen:
  - `subsample`: 0.8
  - `num_leaves`: 100
  - `n_estimators`: 300
  - `max_depth`: 7
  - `learning_rate`: 0.1
  - `colsample_bytree`: 0.9

### Evaluación del Modelo
- Se evalúa el modelo en el conjunto de prueba, obteniendo:
  - **Accuracy**: 97.75%
  - **F1-Score ponderado**: 97.75%
- Se genera un informe de clasificación detallado y una matriz de confusión.

## Resultados
- El modelo entrenado alcanzó un **accuracy del 97.75%** en el conjunto de prueba.
- La clasificación de las categorías de rating mostró un excelente desempeño, con valores altos de **precisión, recall y f1-score** para todas las clases.

## Requisitos
### Librerías utilizadas:
- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`

## Cómo Usar
1. Asegúrate de tener el archivo de datos **Juegorawg_limpio.csv** en la ruta especificada.
2. Ejecuta las celdas del notebook en orden para reproducir los resultados.
3. Modifica los hiperparámetros o las características seleccionadas para experimentar con el modelo.

## Conclusión
Este notebook demuestra un flujo completo de **preprocesamiento, ingeniería de características y clasificación** utilizando un modelo avanzado como **LightGBM**. Los resultados obtenidos son prometedores y pueden ser mejorados con **más datos o ajustes adicionales**.

---
