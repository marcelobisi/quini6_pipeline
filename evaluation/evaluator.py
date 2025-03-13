# evaluation/evaluator.py
import torch
from quini6_pipeline.training import metrics # Importar métricas del módulo training

def evaluar_modelo(modelo, X_test, y_test):
    modelo.eval()
    with torch.no_grad():
        test_outputs = modelo(X_test)
        test_loss = metrics.mse_loss(test_outputs, y_test) # Usar la función de pérdida como métrica
        test_mae = metrics.mae_metric(test_outputs, y_test) # Usar la métrica MAE
    return test_loss, test_mae