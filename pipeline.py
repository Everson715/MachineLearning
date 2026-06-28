import subprocess
import sys

def run_script(script_name):
    print(f"Executando {script_name}...")
    try:
        # Usa o mesmo executável python do ambiente atual
        subprocess.run([sys.executable, script_name], check=True)
        print(f"{script_name} concluído com sucesso.\n")
    except subprocess.CalledProcessError:
        print(f"Erro ao executar {script_name}. Abortando o pipeline.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Erro: O ficheiro {script_name} não foi encontrado. Abortando o pipeline.")
        sys.exit(1)

def main():
    scripts = [
        "prepare_data.py",
        "train_svm.py",
        "train_rf.py",
        "train_mlp.py",
        "main.py"
    ]
    
    print("Iniciando a esteira de Machine Learning...\n")
    for script in scripts:
        run_script(script)
    print("Esteira concluída com sucesso!")

if __name__ == "__main__":
    main()