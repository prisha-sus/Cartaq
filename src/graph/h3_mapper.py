import h3
import pandas as pd
import numpy as np


class H3Mapper:
    def __init__(self, resolution=8):
        self.resolution = resolution

    def add_h3_indices(self, df, lat_col='lat', lon_col='lon'):
        """
        Adds an 'h3_index' column to the DataFrame based on lat/lon columns.
        """
        if df.empty:
            return df

        def get_h3(row):
            if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
                return None
            return h3.latlng_to_cell(row[lat_col], row[lon_col], self.resolution)

        df['h3_index'] = df.apply(get_h3, axis=1)
        return df

    def get_k_ring(self, h3_index, k=1):
        """Returns the k-ring of an H3 cell."""
        return list(h3.grid_disk(h3_index, k))

    def build_adjacency_matrix(self, h3_indices):
        """
        Unweighted k-ring adjacency (k=1).
        Returns edge_index array of shape (2, num_edges).
        """
        index_to_id = {h3_idx: i for i, h3_idx in enumerate(h3_indices)}
        edges = []
        for h3_idx in h3_indices:
            u = index_to_id[h3_idx]
            for neighbor in self.get_k_ring(h3_idx, k=1):
                if neighbor in index_to_id and neighbor != h3_idx:
                    edges.append([u, index_to_id[neighbor]])
        if not edges:
            return np.empty((2, 0), dtype=np.int64)
        return np.array(edges, dtype=np.int64).T

    def build_wind_weighted_edges(
        self,
        h3_indices: list,
        wind_dir_deg: float,
        wind_speed_kmh: float,
    ) -> tuple:
        """
        Build directed edges weighted by wind transport.

        For each adjacent pair (src -> dst):
            weight = max(0, cos(wind_direction - bearing(src->dst))) * wind_speed

        Positive weight = wind blows FROM src TOWARD dst.
        Zero            = crosswind or headwind (no transport on this edge).

        Args:
            h3_indices:    ordered list of H3 cell IDs
            wind_dir_deg:  met convention: direction FROM which wind blows
                           e.g. 270 = westerly wind blowing eastward
            wind_speed_kmh: wind speed in km/h

        Returns:
            edge_index  : np.ndarray [2, num_edges]
            edge_weight : np.ndarray [num_edges]  (values in [0, 1])
        """
        from model.physics_loss import compute_wind_edge_weights
        return compute_wind_edge_weights(h3_indices, wind_dir_deg, wind_speed_kmh)
