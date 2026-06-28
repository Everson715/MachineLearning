# MachineLearning

Este repositório contém um projeto de Machine Learning focado na classificação binária de renda utilizando o dataset Adult da UCI Machine Learning Repository. O objetivo é prever se a renda de um indivíduo excede 50 mil dólares anuais com base em dados de censo demográfico.

## Configuração do Ambiente

Siga os passos abaixo para configurar o ambiente virtual e instalar as dependências necessárias.

1. **Criar o ambiente virtual:**
   ```bash
   python -m venv .venv
   ```

2. **Ativar o ambiente virtual:**
   - No Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - No Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Instalar as dependências:**
   ```bash
   pip install streamlit matplotlib seaborn scikit-learn pandas numpy ucimlrepo
   ```

## Como Executar

### 1. Executar o Pipeline Completo

Para executar toda a esteira de processamento de dados, treino dos modelos (SVM, Random Forest, MLP) e consolidação dos resultados, basta correr o script de pipeline na raiz do projeto:

```bash
python pipeline.py
```

A ordem de execução definida no pipeline é:
1. `prepare_data.py`: Baixa o dataset, faz o split, aplica o `ColumnTransformer` e gera `data_processed.npz`.
2. `train_svm.py`: Treina o modelo SVM e salva métricas em `results_svm.json`.
3. `train_rf.py`: Treina o modelo Random Forest e salva métricas em `results_rf.json`.
4. `train_mlp.py`: Treina o modelo MLP e salva métricas em `results_mlp.json`.
5. `main.py`: Lê todos os ficheiros JSON e gera um relatório consolidado com a comparação dos modelos em `relatorio_desempenho.json`.

> **Nota:** Todos os ficheiros de resultados incluem agora a `accuracy`, `train_accuracy` e a `confusion_matrix`.

### 2. Testar os Resultados

Após executar o pipeline, pode correr o script de testes para garantir que todos os ficheiros de saída contêm o formato e as métricas esperadas:

```bash
python test_models.py
```

### 3. Inicializar o Dashboard Interativo

O projeto também inclui um dashboard interativo desenvolvido com Streamlit. Para visualizar os dados e o desempenho dos modelos de forma gráfica, execute o seguinte comando:

```bash
streamlit run app_dashboard.py
```

O Dashboard foi projetado com layout expandido e possui 3 abas principais:
- **Visão Geral e Comparativo**: Apresenta uma tabela de métricas consolidada e um gráfico comparativo de Acurácia de Treino vs. Teste para facilitar a identificação de *Overfitting*.
- **Matrizes de Confusão**: Exibe lado a lado os mapas de calor (*heatmaps*) das matrizes de confusão de cada modelo, avaliando o desempenho nas classes individuais (`<=50K` vs `>50K`).
- **Relatório por Modelo**: Permite a seleção de um modelo específico para visualizar em detalhe o relatório de classificação completo, com Precisão, Recall e F1-Score separados por classe.
