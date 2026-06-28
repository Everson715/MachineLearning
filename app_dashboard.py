import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração da página (deve ser a primeira chamada do Streamlit)
st.set_page_config(page_title="Dashboard de Machine Learning", layout="wide")

def load_data():
    try:
        with open("relatorio_desempenho.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("comparacao_modelos", {})
    except FileNotFoundError:
        st.error("Erro: O ficheiro 'relatorio_desempenho.json' não foi encontrado. Por favor, execute o pipeline primeiro.")
        return {}
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return {}

def main():
    st.title("Dashboard Comparativo de Modelos: Previsão de Renda")
    st.markdown("Análise baseada no dataset Adult da UCI Machine Learning Repository.")
    st.markdown("---")
    
    dados = load_data()
    
    if not dados:
        return
        
    # Inicializando as 3 abas
    tab1, tab2, tab3 = st.tabs(["Visão Geral e Comparativo", "Matrizes de Confusão", "Relatório por Modelo"])
    
    # =========================================================================
    # Aba 1: Visão Geral e Comparativo
    # =========================================================================
    with tab1:
        st.header("Visão Geral e Comparativo")
        
        # Preparando os dados para a tabela
        modelos_list = []
        for model_name, metrics in dados.items():
            modelos_list.append({
                "Modelo": model_name.upper(),
                "Acurácia (Treino)": metrics.get("train_accuracy", 0.0),
                "Acurácia (Teste)": metrics.get("accuracy", 0.0),
                "Precisão (Macro)": metrics.get("macro avg", {}).get("precision", 0.0),
                "F1-Score (Macro)": metrics.get("macro avg", {}).get("f1-score", 0.0)
            })
            
        df_comparativo = pd.DataFrame(modelos_list)
        
        st.subheader("Tabela Comparativa de Métricas")
        # Formatando para exibição: acurácia em percentagem e restantes com 4 casas decimais
        st.dataframe(df_comparativo.style.format({
            "Acurácia (Treino)": "{:.2%}",
            "Acurácia (Teste)": "{:.2%}",
            "Precisão (Macro)": "{:.4f}",
            "F1-Score (Macro)": "{:.4f}"
        }), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Comparativo de Acurácia: Treino vs Teste (Avaliação de Overfitting)")
        
        # Gráfico de barras lado a lado (Matplotlib)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(len(df_comparativo['Modelo']))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, df_comparativo['Acurácia (Treino)'], width, label='Treino', color='#4c72b0')
        rects2 = ax.bar(x + width/2, df_comparativo['Acurácia (Teste)'], width, label='Teste', color='#dd8452')
        
        ax.set_ylabel('Acurácia')
        ax.set_title('Acurácia de Treino vs Teste por Modelo')
        ax.set_xticks(x)
        ax.set_xticklabels(df_comparativo['Modelo'])
        ax.legend()
        
        # Adicionando rótulos aos dados das barras
        ax.bar_label(rects1, padding=3, fmt='%.3f')
        ax.bar_label(rects2, padding=3, fmt='%.3f')
        
        # Ajustando a escala Y (de 0 até 1) com margem
        ax.set_ylim(0, 1.1)
        
        fig.tight_layout()
        st.pyplot(fig)
        
    # =========================================================================
    # Aba 2: Matrizes de Confusão
    # =========================================================================
    with tab2:
        st.header("Matrizes de Confusão (Dados de Teste)")
        st.markdown("Comparativo visual dos erros e acertos de cada modelo na classificação das classes.")
        
        # Colocando os gráficos lado a lado
        cols = st.columns(len(dados))
        class_names = ['<=50K', '>50K']
        
        for idx, (model_name, metrics) in enumerate(dados.items()):
            with cols[idx]:
                st.subheader(f"{model_name.upper()}")
                cm = np.array(metrics.get("confusion_matrix", []))
                
                if cm.size > 0:
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                                xticklabels=class_names, yticklabels=class_names)
                    
                    ax_cm.set_xlabel('Predito')
                    ax_cm.set_ylabel('Real')
                    st.pyplot(fig_cm)
                else:
                    st.warning("Matriz de confusão não disponível no JSON.")
                    
    # =========================================================================
    # Aba 3: Relatório por Modelo
    # =========================================================================
    with tab3:
        st.header("Relatório Detalhado por Modelo")
        
        model_options = list(dados.keys())
        selected_model = st.selectbox("Selecione um modelo para visualizar as métricas por classe:", model_options)
        
        if selected_model:
            st.markdown(f"### Detalhes de Desempenho do Modelo: **{selected_model.upper()}**")
            
            metrics = dados[selected_model]
            class_names_mapping = {"0": "<=50K", "1": ">50K"}
            
            # Extraindo as classes "0" e "1"
            class_metrics = []
            for class_label in ["0", "1"]:
                if class_label in metrics:
                    cls_data = metrics[class_label]
                    class_metrics.append({
                        "Classe": class_names_mapping.get(class_label, class_label),
                        "Precisão": cls_data.get("precision", 0.0),
                        "Recall": cls_data.get("recall", 0.0),
                        "F1-Score": cls_data.get("f1-score", 0.0),
                        "Suporte": cls_data.get("support", 0)
                    })
                    
            if class_metrics:
                df_class = pd.DataFrame(class_metrics)
                st.table(df_class.style.format({
                    "Precisão": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-Score": "{:.4f}",
                    "Suporte": "{:,.0f}"
                }))
            else:
                st.warning("Métricas por classe não encontradas para este modelo.")
                
            st.markdown("---")
            st.markdown("#### Métricas Globais")
            
            # Exibir médias globais
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Acurácia (Teste)", f"{metrics.get('accuracy', 0.0):.2%}")
            with col_b:
                st.metric("F1-Score (Macro)", f"{metrics.get('macro avg', {}).get('f1-score', 0.0):.4f}")
            with col_c:
                st.metric("F1-Score (Ponderado)", f"{metrics.get('weighted avg', {}).get('f1-score', 0.0):.4f}")

if __name__ == "__main__":
    main()