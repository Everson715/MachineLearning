# Projeto de Machine Learning: Comparação de Modelos

Este projeto consiste em um pipeline completo de Machine Learning para treinar e comparar diferentes modelos de classificação. Os dados utilizados vêm do dataset **Adult** do repositório UCI.

## Estrutura do Projeto

O projeto é dividido em diversos scripts Python, cada um com uma responsabilidade específica no pipeline:

* **`prepare_data.py`**: Faz o download do dataset UCI, limpa e pré-processa os dados (One-Hot Encoding, StandardScaler e Label Encoding), além de dividir em conjunto de treino e teste. O resultado é salvo no arquivo `data_processed.npz`.
* **`train_svm.py`**: Treina um modelo Support Vector Machine (SVM) com os dados processados e exporta o relatório e matriz de confusão para `results_svm.json`.
* **`train_rf.py`**: Treina um modelo Random Forest (RF) e exporta os resultados para `results_rf.json`.
* **`train_mlp.py`**: Treina uma Rede Neural Multi-Layer Perceptron (MLP) e exporta os resultados para `results_mlp.json`.
* **`main.py`**: Lê todos os arquivos `json` gerados pelas etapas de treinamento e os consolida em um único arquivo chamado `relatorio_desempenho.json`.
* **`pipeline.py`**: Script automatizado que executa todo o fluxo de trabalho descrito acima na ordem correta.

## Como Executar

A forma mais simples de rodar todo o projeto é utilizando o script de pipeline, que executa automaticamente a preparação dos dados, o treinamento de todos os modelos e a geração do relatório final consolidado.

```bash
python pipeline.py
```

### Execução Passo a Passo (Opcional)
Se preferir rodar as etapas separadamente:

1. **Preparo dos dados**:
   ```bash
   python prepare_data.py
   ```
2. **Treinamento dos Modelos**:
   ```bash
   python train_svm.py
   python train_rf.py
   python train_mlp.py
   ```
3. **Consolidação do Relatório**:
   ```bash
   python main.py
   ```

## Dependências

As bibliotecas principais utilizadas neste projeto são:
- `numpy`
- `scikit-learn`
- `ucimlrepo`

## Saídas (Outputs)
- `data_processed.npz`: Dados prontos para o treinamento.
- `results_*.json`: Métricas individuais de cada modelo (contendo também a matriz de confusão).
- `relatorio_desempenho.json`: Arquivo consolidado com a comparação final entre todos os algoritmos.
