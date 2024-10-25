import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle
from statsbombpy import sb

# Obtener datos de StatsBomb usando statsbombpy
competitions = sb.competitions()
copa_america = competitions[(competitions['competition_id'] == 223) & (competitions['season_id'] == 282)]
euro = competitions[(competitions['competition_id'] == 55) & (competitions['season_id'] == 282)]

# Unir los partidos de ambas competiciones
matches = pd.concat([
    sb.matches(competition_id=copa_america['competition_id'].values[0], season_id=copa_america['season_id'].values[0]),
    sb.matches(competition_id=euro['competition_id'].values[0], season_id=euro['season_id'].values[0])
], ignore_index=True)

# Obtener eventos de los partidos y crear un dataframe
event_data = []

for match_id in matches['match_id']:
    events = sb.events(match_id=match_id)
    for _, event in events.iterrows():
        # Extraer características relevantes
        minute = event.get('minute', 0)
        fatigue = np.random.randint(50, 100)  # Esto se puede ajustar con datos reales si están disponibles
        performance = np.random.randint(0, 100)  # Esto se puede ajustar con datos reales si están disponibles
        substitution_needed = 1 if event['type'] == 'Substitution' else 0

        event_data.append([minute, fatigue, performance, substitution_needed])

# Crear DataFrame
columns = ['minute', 'fatigue', 'performance', 'substitution_needed']
data = pd.DataFrame(event_data, columns=columns)

# Paso 3: Preprocesar los datos y entrenar el modelo
features = ['minute', 'fatigue', 'performance']
target = 'substitution_needed'
X = data[features]
y = data[target]

# Balancear los datos usando RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo entrenado
with open('substitution_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Modelo para predecir el rendimiento del equipo
def calculate_team_performance(players):
    return np.mean([player['performance'] - player['fatigue'] for player in players])

# Guardar la función para calcular el rendimiento del equipo
with open('team_performance_model.pkl', 'wb') as performance_file:
    pickle.dump(calculate_team_performance, performance_file)