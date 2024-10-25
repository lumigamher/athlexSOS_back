from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

# Definir la función para calcular el rendimiento del equipo
def calculate_team_performance(players):
    return np.mean([player['performance'] - player['fatigue'] for player in players])

# Cargar el modelo entrenado para sustitución
with open('substitution_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Habilitar CORS para permitir solicitudes desde diferentes orígenes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/check_substitution', methods=['POST'])
def check_substitution():
    data = request.get_json()
    minute = data.get('minute')
    fatigue = data.get('fatigue')
    performance = data.get('performance')
    
    # Realizar la predicción con el modelo
    input_features = np.array([[minute, fatigue, performance]])
    prediction = model.predict(input_features)
    
    if prediction[0] == 1:
        recommendation = "Recomendación: Realizar sustitución. El modelo indica que es el momento óptimo."
    else:
        recommendation = "No se recomienda una sustitución en este momento."
    
    return jsonify({'recommendation': recommendation})

@app.route('/api/team_performance', methods=['POST'])
def team_performance():
    data = request.get_json()
    players = data.get('players')
    
    # Calcular el rendimiento del equipo
    performance_score = calculate_team_performance(players)
    
    return jsonify({'performance_score': performance_score})

if __name__ == '__main__':
    app.run(debug=True, port=5000)