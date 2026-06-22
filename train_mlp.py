import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregar dados processados
data = np.load('data_processed.npz')

# Configuração do MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50), 
    max_iter=300, 
    random_state=42, 
    early_stopping=True, 
    verbose=True
)

print("Treinando MLP...")
mlp.fit(data['xt'], data['yt'])

# Avaliação
y_pred = mlp.predict(data['xv'])
cm = confusion_matrix(data['yv'], y_pred)

report = classification_report(data['yv'], y_pred, output_dict=True)
report['confusion_matrix'] = cm.tolist()

with open('results_mlp.json', 'w') as f:
    json.dump(report, f, indent=4)

print("MLP concluído. Resultados em 'results_mlp.json'.")