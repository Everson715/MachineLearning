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

## Preparando o Ambiente e Executando

Recomenda-se o uso de um ambiente virtual (venv) para evitar conflitos de dependências. Abaixo estão as orientações para preparar o ambiente e rodar o pipeline no Linux e no Windows.

### Em Linux ou macOS

1. **Crie o ambiente virtual (venv)**:
   ```bash
   python3 -m venv .venv
   ```

2. **Ative o ambiente virtual**:
   ```bash
   source .venv/bin/activate
   ```

3. **Instale as dependências** (caso possua um `requirements.txt`):
   ```bash
   pip install numpy scikit-learn ucimlrepo
   ```

4. **Execute o Pipeline Completo**:
   ```bash
   python pipeline.py
   ```

### Em Windows

1. **Crie o ambiente virtual (venv)**:
   ```cmd
   python -m venv .venv
   ```

2. **Ative o ambiente virtual**:
   - Prompt de Comando (CMD):
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```

3. **Instale as dependências**:
   ```cmd
   pip install numpy scikit-learn ucimlrepo
   ```

4. **Execute o Pipeline Completo**:
   ```cmd
   python pipeline.py
   ```

### Execução Passo a Passo (Opcional)
Se preferir rodar as etapas separadamente, com o ambiente virtual ativado, execute:

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
