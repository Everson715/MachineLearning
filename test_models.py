import unittest
import json
import os

class TestModelResults(unittest.TestCase):
    
    def check_result_file(self, filename):
        # Verifica se o ficheiro existe
        self.assertTrue(os.path.exists(filename), f"O ficheiro {filename} não foi encontrado.")
        
        # Carrega o JSON
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Verifica se contém as chaves obrigatórias
        required_keys = ['accuracy', 'train_accuracy', 'confusion_matrix']
        for key in required_keys:
            self.assertIn(key, data, f"A chave obrigatória '{key}' está a faltar no ficheiro {filename}.")
            
    def test_svm_results(self):
        self.check_result_file("results_svm.json")
        
    def test_rf_results(self):
        self.check_result_file("results_rf.json")
        
    def test_mlp_results(self):
        self.check_result_file("results_mlp.json")

if __name__ == "__main__":
    unittest.main()
