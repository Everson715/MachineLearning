import json
import os

def gerar_relatorio_final():
    # Inclua 'mlp' aqui, já que agora temos o script dele também
    modelos = ['svm', 'rf', 'mlp']
    relatorio_consolidado = {"comparacao_modelos": {}}

    for nome in modelos:
        arquivo = f'results_{nome}.json'
        if os.path.exists(arquivo):
            with open(arquivo, 'r') as f:
                # O json.load lerá automaticamente a nova chave 'confusion_matrix'
                dados = json.load(f)
                relatorio_consolidado["comparacao_modelos"][nome.upper()] = dados
        else:
            print(f"Aviso: {arquivo} não encontrado.")

    with open('relatorio_desempenho.json', 'w', encoding='utf-8') as f:
        json.dump(relatorio_consolidado, f, indent=4, ensure_ascii=False)
    
    print("Relatório 'relatorio_desempenho.json' atualizado com sucesso!")

if __name__ == "__main__":
    gerar_relatorio_final()