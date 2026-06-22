import numpy as np
import json
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

print("Lendo dados processados...")
data = np.load('data_processed.npz')

print("Treinando o modelo SVM...")
# Aqui garantimos o uso do algoritmo SVM (SVC)
model = SVC(kernel='rbf', random_state=42, max_iter=10000)
model.fit(data['xt'], data['yt'])

print("Calculando acurácia de treinamento...")
train_acc = model.score(data['xt'], data['yt'])

y_pred = model.predict(data['xv'])

report = classification_report(data['yv'], y_pred, output_dict=True)
report['train_accuracy'] = float(train_acc)
report['confusion_matrix'] = confusion_matrix(data['yv'], y_pred).tolist()

# Certifique-se de que o nome do arquivo aqui seja results_svm.json
with open('results_svm.json', 'w') as f: 
    json.dump(report, f, indent=4)

print("SVM concluído com sucesso!")