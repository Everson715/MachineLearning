import subprocess

# A ordem correta é: Preparar -> Treinar -> Consolidar
scripts = ['prepare_data.py', 'train_svm.py', 'train_rf.py', 'train_mlp.py', 'main.py']

for script in scripts:
    print(f"Executando {script}...")
    subprocess.run(['python', script], check=True)

print("Pipeline completo! Relatório final disponível.")