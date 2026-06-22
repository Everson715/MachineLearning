import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("Lendo dados processados...")
data = np.load('data_processed.npz')

print("Treinando o modelo de Rede Neural (MLP)...")
model = MLPClassifier(
    hidden_layer_sizes=(100, 50), 
    max_iter=300, 
    random_state=42, 
    early_stopping=True
)
model.fit(data['xt'], data['yt'])

# 1. Calcular Acurácia de Treino
print("Calculando acurácia de treinamento...")
train_acc = model.score(data['xt'], data['yt'])

# 2. Realizar predições no teste
y_pred = model.predict(data['xv'])

# 3. Calcular relatório de classificação e matriz de confusão
report = classification_report(data['yv'], y_pred, output_dict=True)
report['train_accuracy'] = float(train_acc)
report['confusion_matrix'] = confusion_matrix(data['yv'], y_pred).tolist()

# 4. Salvar resultados
with open('results_mlp.json', 'w') as f: 
    json.dump(report, f, indent=4)

print("MLP concluído com sucesso!")