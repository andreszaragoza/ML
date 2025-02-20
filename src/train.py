import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


file_path = 'data\processed\Juegorawg_limpio.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"El archivo {file_path} no se encontró. Verifica la ruta y asegúrate de que el archivo existe.")

df = pd.read_csv(file_path)
df.copy()


df['released'] = pd.to_datetime(df['released'])
df['updated'] = pd.to_datetime(df['updated'])


df['playtime_log'] = df['playtime'].apply(lambda x: np.log1p(x))


df = pd.get_dummies(df, columns=['main_genre', 'metacritic_category'], drop_first=True)


df = df.select_dtypes(include=[np.number])


scaler = StandardScaler()
num_vars = ['rating', 'metacritic', 'playtime_log']
df[num_vars] = scaler.fit_transform(df[num_vars])


X = df.drop(columns=['rating'])
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}")
print(f"R² Score: {r2}")


model_path = 'C:/Users/anoni/ML/models/trained/new_model_random.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(rf_model, model_path)
print(f"Modelo guardado en: {model_path}")
