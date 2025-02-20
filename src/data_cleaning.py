import ast
import pandas as pd
from collections import Counter

def convert_to_json(data):
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        return None

def unnest_column(df, column_name):
    """
    Convierte cadenas JSON-like, explota listas y normaliza diccionarios en una columna específica.
    """
    df[column_name] = df[column_name].dropna().apply(convert_to_json)

    exploded = df[column_name].explode().dropna()

    unnested_column = pd.json_normalize(exploded) 

    return unnested_column


def extract_genre_names(genres):
    """
    Extraer el nombre de cada género
    """
    try:
        genres_list = ast.literal_eval(genres)
        return [genre['name'] for genre in genres_list]
    except (ValueError, TypeError):
        return []

def extract_tag_names(tag_string):
    """
    Extraer el nombre de cada etiqueta
    """
    try:
        if pd.notnull(tag_string):
            return [tag['name'] for tag in ast.literal_eval(tag_string)]
        else:
            return []
    except (ValueError, SyntaxError, TypeError):
        return []
    
def extract_store_names(stores):
    try:
        stores = ast.literal_eval(stores)
        return [store['store']['name'] for store in stores]
    except Exception:
        return []
    
def extract_main_genre(genre_str):
    # Si el valor es None o np.nan
    if genre_str is None or (isinstance(genre_str, float) and pd.isna(genre_str)):
        return 'Unknown'
    
    # Si ya es una lista
    if isinstance(genre_str, list):
        return genre_str[0]['name'] if genre_str and isinstance(genre_str[0], dict) else 'Unknown'
    
    # Si es una cadena, intentar convertirla a lista
    if isinstance(genre_str, str):
        try:
            genres_list = ast.literal_eval(genre_str)
            if isinstance(genres_list, list) and genres_list and isinstance(genres_list[0], dict):
                return genres_list[0].get('name', 'Unknown')
        except (ValueError, SyntaxError):
            return 'Unknown'
    
    return 'Unknown'

def categorize_metacritic(score):
    if score >= 80:
        return "Alta"
    elif score >= 50:
        return "Media"
    else:
        return "Baja"
    

def preprocess_column(df, column_name):
    """
    
    """
    if df[column_name].dtype == 'object' and df[column_name].apply(lambda x: isinstance(x, list)).any():
        return df[column_name].explode().dropna().nunique()
    else:
        return df[column_name].nunique()