import json
import os

def load_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def main():
    models = {
        "svm": "results_svm.json",
        "rf": "results_rf.json",
        "mlp": "results_mlp.json"
    }
    
    comparacao_modelos = {}
    
    for model_name, filename in models.items():
        data = load_results(filename)
        if data:
            comparacao_modelos[model_name] = data
        else:
            print(f"Aviso: {filename} não foi encontrado.")
            
    final_report = {
        "comparacao_modelos": comparacao_modelos
    }
    
    with open("relatorio_desempenho.json", 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
        
    print("Relatório de desempenho gerado com sucesso: relatorio_desempenho.json")

if __name__ == "__main__":
    main()