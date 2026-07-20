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
        """
        Returns the k-ring of an H3 cell.
        """
        return list(h3.grid_disk(h3_index, k))
        
    def build_adjacency_matrix(self, h3_indices):
        """
        Builds a sparse adjacency list (edge_index) for the given unique H3 indices.
        Unweighted k-ring adjacency (k=1).
        Returns edge_index array of shape (2, num_edges).
        """
        index_to_id = {h3_idx: i for i, h3_idx in enumerate(h3_indices)}
        
        edges = []
        for h3_idx in h3_indices:
            neighbors = self.get_k_ring(h3_idx, k=1)
            u = index_to_id[h3_idx]
            for neighbor in neighbors:
                if neighbor in index_to_id and neighbor != h3_idx:
                    v = index_to_id[neighbor]
                    edges.append([u, v])
                    
        if not edges:
            return np.empty((2, 0), dtype=np.int64)
            
        # Convert to numpy array of shape (2, E)
        edge_index = np.array(edges).T
        return edge_index
