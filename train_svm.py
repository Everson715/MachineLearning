import numpy as np
import json
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data = np.load('data_processed.npz')
model = SVC(kernel='rbf', random_state=42, max_iter=10000)
model.fit(data['xt'], data['yt'])

# Realiza a predição
y_pred = model.predict(data['xv'])

# Calcula métricas e matriz
report = classification_report(data['yv'], y_pred, output_dict=True)
report['confusion_matrix'] = confusion_matrix(data['yv'], y_pred).tolist()

with open('results_svm.json', 'w') as f: 
    json.dump(report, f, indent=4)
print("SVM concluído com matriz de confusão.")