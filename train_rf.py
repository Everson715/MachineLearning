import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("Lendo dados processados...")
data = np.load('data_processed.npz')

print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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
with open('results_rf.json', 'w') as f: 
    json.dump(report, f, indent=4)

print("Random Forest concluído com sucesso!")