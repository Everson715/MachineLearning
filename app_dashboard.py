import streamlit as st
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Dashboard de Avaliação de Modelos - Projeto IA")

# Carregar dados
@st.cache_data
def load_data():
    with open('relatorio_desempenho.json', 'r', encoding='utf-8') as f:
        return json.load(f)['comparacao_modelos']

data = load_data()

# 1. Tabela Comparativa (Requisito da página 8 da apresentação)
st.header("Comparativo de Desempenho")
resumo = []
for modelo, perf in data.items():
    resumo.append({
        "Modelo": modelo,
        "Acurácia": f"{perf['accuracy']*100:.2f}%",
        "Precisão (Macro)": f"{perf['macro avg']['precision']:.4f}",
        "F1-Score (Macro)": f"{perf['macro avg']['f1-score']:.4f}"
    })

st.table(pd.DataFrame(resumo))

# 2. Matrizes de Confusão (Requisito da página 9 da apresentação)
st.header("Matrizes de Confusão")
cols = st.columns(len(data))
for i, (modelo, perf) in enumerate(data.items()):
    with cols[i]:
        st.subheader(modelo)
        cm = perf['confusion_matrix']
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
        st.pyplot(fig)