import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

file_path = 'data\processed\Juegorawg_limpio.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"El archivo {file_path} no se encontró. Verifica la ruta y asegúrate de que el archivo existe.")

df = pd.read_csv(file_path)
df.copy()

# Verificar si ya existe la columna rating_category, si no, crearla
if 'rating_category' not in df.columns:
    rating_bins = [0, 2.5, 3.5, 5]
    rating_labels = ['Bajo', 'Medio', 'Alto']
    df['rating_category'] = pd.cut(df['rating'], bins=rating_bins, labels=rating_labels)
    
    # Imputar valores nulos en rating_category
    df.loc[df['rating_category'].isnull() & (df['rating'] <= 2.5), 'rating_category'] = 'Bajo'
    df.loc[df['rating_category'].isnull() & (df['rating'] > 2.5) & (df['rating'] <= 3.5), 'rating_category'] = 'Medio'
    df.loc[df['rating_category'].isnull() & (df['rating'] > 3.5), 'rating_category'] = 'Alto'

# Crear características derivadas
df['rating_to_count_ratio'] = df['rating'] / (df['ratings_count'] + 1)  # +1 para evitar división por cero
df['reviews_to_ratings_ratio'] = df['reviews_count'] / (df['ratings_count'] + 1)

# Características importantes según lo mencionado
important_features = [
    'rating_to_count_ratio',
    'reviews_count',
    'ratings_count',
    'reviews_to_ratings_ratio',
    'added'
]


X = df[important_features]
y = df['rating_category']

# Codificar la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo GradientBoosting
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_clf.fit(X_train_scaled, y_train)

# Crear una explicación manual del modelo basada en la importancia de características
feature_importance = pd.DataFrame({
    'Feature': important_features,
    'Importance': gb_clf.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Graficar importancia de características
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Importancia de Características en el Modelo Gradient Boosting')
plt.tight_layout()
plt.savefig('feature_importance_final.png')
plt.close()

# Guardar el modelo y componentes necesarios para la inferencia
model_components = {
    'model': gb_clf,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'important_features': important_features
}

model_path = 'C:/Users/anoni/ML/models/trained/Mi_modelo.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(gb_clf, model_path)
print(f"Modelo guardado en: {model_path}")