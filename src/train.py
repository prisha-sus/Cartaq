import torch
import numpy as np
import pandas as pd
from model.a3tgcn import A3TGCNModel
from model.baseline import PersistenceBaseline
import json
import os

def load_dummy_graph_data():
    """
    Mock function to generate random graph data for testing.
    In a real scenario, this would load the output from src/graph/graph_builder.py
    """
    num_nodes = 50
    num_features = 4  # e.g., PM2.5, temp, humidity, wind speed
    periods = 12      # e.g., past 12 hours
    batch_size = 16
    
    # Generate random features: [batch_size, num_nodes, num_features, periods]
    x = torch.randn(batch_size, num_nodes, num_features, periods)
    
    # Generate random labels for 3 horizons: [batch_size, num_nodes, 3]
    y = torch.randn(batch_size, num_nodes, 3)
    
    # Generate random edges: [2, num_edges]
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    return x, y, edge_index

def train_model():
    print("Loading data...")
    x, y, edge_index = load_dummy_graph_data()
    
    # Split into train/val/test (temporal split)
    # Since we have mock batched data, let's just split the batch dim
    train_size = int(0.7 * x.size(0))
    val_size = int(0.15 * x.size(0))
    
    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
    x_test, y_test = x[train_size+val_size:], y[train_size+val_size:]
    
    print("Evaluating baseline...")
    baseline = PersistenceBaseline()
    
    # Current AQI is the last feature of the last period (let's assume feature 0 is AQI)
    current_aqi_test = x_test[:, :, 0, -1]
    y_pred_baseline = baseline.predict(current_aqi_test)
    baseline_rmse = baseline.evaluate(y_test, y_pred_baseline)
    print(f"Baseline RMSE (24h, 48h, 72h): {baseline_rmse.tolist()}")
    
    print("Training A3TGCN...")
    model = A3TGCNModel(node_features=x.size(2), periods=x.size(3), hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train, edge_index)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.sleep = 0
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(x_val, edge_index)
            val_loss = criterion(val_out, y_val)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        
    # Test Evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(x_test, edge_index)
        # Calculate RMSE per horizon
        test_rmse = torch.sqrt(torch.mean((test_out - y_test)**2, dim=(0, 1)))
        
    print(f"A3TGCN Test RMSE (24h, 48h, 72h): {test_rmse.tolist()}")
    
    metrics = {
        "baseline_rmse": baseline_rmse.tolist(),
        "a3tgcn_rmse": test_rmse.tolist(),
        "graph_construction_latency_p50": 1.2, # Mock
        "graph_construction_latency_p95": 2.5  # Mock
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics.json")
    
if __name__ == "__main__":
    train_model()
