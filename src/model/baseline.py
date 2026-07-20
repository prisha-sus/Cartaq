import numpy as np

class PersistenceBaseline:
    def __init__(self):
        pass
        
    def predict(self, current_aqi):
        """
        Naive forecast: AQI(t+h) = AQI(t)
        current_aqi shape: (num_nodes,) or (batch, num_nodes)
        Returns predictions for +24h, +48h, +72h.
        """
        # We just duplicate the current AQI for all horizons
        # Assuming horizons are 3 (24, 48, 72)
        if isinstance(current_aqi, np.ndarray):
            # If shape is (N,), make it (N, 3)
            # If shape is (B, N), make it (B, N, 3)
            return np.stack([current_aqi, current_aqi, current_aqi], axis=-1)
        import torch
        if isinstance(current_aqi, torch.Tensor):
            return torch.stack([current_aqi, current_aqi, current_aqi], dim=-1)
            
        raise ValueError("Unsupported data type")
        
    def evaluate(self, y_true, y_pred):
        """
        Calculates RMSE per horizon.
        y_true, y_pred expected shape: (..., num_horizons) where num_horizons=3
        """
        if isinstance(y_true, np.ndarray):
            mse = np.mean((y_true - y_pred)**2, axis=tuple(range(y_true.ndim - 1)))
            return np.sqrt(mse)
            
        import torch
        if isinstance(y_true, torch.Tensor):
            # Average over all dims except the last one (horizons)
            dim = tuple(range(y_true.ndim - 1))
            mse = torch.mean((y_true - y_pred)**2, dim=dim)
            return torch.sqrt(mse)
            
        raise ValueError("Unsupported data type")
