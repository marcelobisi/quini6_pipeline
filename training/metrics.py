# training/metrics.py
import torch
import numpy as np

def mse_loss(outputs, targets):
    criterion = nn.MSELoss()
    return criterion(outputs, targets)

def mae_metric(outputs, targets): # Ejemplo de MAE (adaptar a tu caso)
    return torch.mean(torch.abs(outputs - targets)).item()

# ... (Otras métricas personalizadas que definas) ...