# Procesado de Datos para Clasificación

Este notebook contiene el procesamiento de datos para un modelo de clasificación. Se incluyen técnicas de limpieza, balanceo de clases y transformación de variables.

## Contenido del Notebook

### Preprocesamiento
- Crear categorías de rating para clasificación
- Verificar valores nulos en `rating_category`
- Imputar los valores nulos en `rating_category`
  - Se usaron los valores de rating normal

### Trabajo con Características
- Enfoque en características de clustering
- Aplicación de clustering con **K-means**
  - Se definieron 5 clusters para mejor segmentación
- Creación de variables dummy para los clusters
- Codificación de `main_genre` usando one-hot encoding

### Preparación y Entrenamiento del Modelo
- Combinación de modelos y aplicación de balanceo de clases
- **Selección de características** para mejorar predicciones
- Entrenamiento del modelo con **Gradient Boosting**
- Evaluación del modelo mediante métricas clave

## Resumen de Resultados

### Logros del Modelo de Clasificación Mejorado

#### 1. Predicción precisa de categorías de rating
- **94.4% de exactitud** con el modelo **Gradient Boosting**
- Clasificación efectiva en tres categorías de rating: **Bajo, Medio y Alto** con alta precisión

#### 2. Rendimiento por categoría
- **Categoría "Bajo"**: 100% de precisión, 96% de recall
- **Categoría "Medio"**: 86% de precisión, 93% de recall
- **Categoría "Alto"**: 90% de precisión, 90% de recall

#### 3. Aplicaciones prácticas
- **Predicción de calidad**: Estimación de la calidad de un juego antes de su lanzamiento
- **Identificación de factores de éxito**: Detección de características clave para un rating alto
- **Segmentación de mercado**: Clasificación de juegos en diferentes segmentos de calidad

### Conclusión
Este modelo proporciona una herramienta efectiva para clasificar y predecir el éxito de videojuegos basado en múltiples características. Puede ser utilizado para análisis de tendencias, desarrollo de nuevos títulos y estrategias de marketing.

---
