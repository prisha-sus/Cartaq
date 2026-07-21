"""
OpenStreetMap (OSM) Land-Use & Road Network Client
Fetches industrial zones and road density via the Overpass API.

Why it matters:
  These are STATIC node features in the graph - they don't change hour-to-hour.
  - Road network density -> proxy for traffic emissions
  - Industrial zones -> proxy for stationary emitter locations
  Both are used as "treatment" candidates in the DoWhy DAG.

Free API: https://overpass-api.de/api/interpreter
No API key required (subject to rate limits).

Usage:
    client = OSMClient(bbox="73.7,18.4,74.0,18.7")
    roads_df = client.fetch_road_network()
    industry_df = client.fetch_industrial_zones()

Output is mapped to H3 hexes and saved as static node feature parquet files.
"""

import os
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.h3_mapper import H3Mapper

# Multiple Overpass mirrors - tried in order until one succeeds
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Headers that prevent 406 Not Acceptable from strict Overpass servers
OVERPASS_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept":       "application/json",
    "User-Agent":   "CartaqResearch/1.0 (academic air quality project)",
}


class OSMClient:
    """
    Fetches road network density and industrial land use from OpenStreetMap
    via the Overpass API. Results are aggregated to H3 hexagonal cells.
    """

    def __init__(
        self,
        bbox: str = "73.7,18.4,74.0,18.7",
        resolution: int = 8,
    ):
        # bbox format for Overpass: south,west,north,east
        parts = [float(x) for x in bbox.split(",")]
        # incoming format: west,south,east,north
        self.west, self.south, self.east, self.north = parts
        self.overpass_bbox = f"{self.south},{self.west},{self.north},{self.east}"
        self.resolution     = resolution
        self.mapper         = H3Mapper(resolution)

    def _query(self, ql_body: str) -> dict:
        """Execute an Overpass QL query, trying each mirror until one works."""
        query = f"[out:json][timeout:60];\n{ql_body}\nout center;"
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(
                    endpoint,
                    data={"data": query},
                    headers=OVERPASS_HEADERS,
                    timeout=90,
                )
                if resp.status_code == 200:
                    return resp.json()
                else:
                    print(f"  Overpass error {resp.status_code} on {endpoint}: {resp.text[:120]}")
            except Exception as e:
                print(f"  Overpass request failed on {endpoint}: {e}")
        # All mirrors failed
        return {"elements": []}

    def fetch_industrial_zones(self) -> pd.DataFrame:
        """
        Fetch industrial + commercial land-use polygons.
        These are mapped to H3 hexes as static indicators.

        Returns DataFrame: lat, lon, osm_id, landuse_type
        """
        print(f"Fetching OSM industrial zones for bbox {self.overpass_bbox}...")
        ql = f"""
        (
          way["landuse"~"industrial|commercial|retail|port"]({self.overpass_bbox});
          relation["landuse"~"industrial|commercial|port"]({self.overpass_bbox});
        );
        """
        data = self._query(ql)
        rows = []
        for el in data.get("elements", []):
            c = el.get("center", {})
            if not c:
                continue
            rows.append({
                "lat":          c.get("lat"),
                "lon":          c.get("lon"),
                "osm_id":       el.get("id"),
                "landuse_type": el.get("tags", {}).get("landuse", "industrial"),
            })
        if not rows:
            print("  No industrial/commercial zones found (Overpass returned no elements).")
            return pd.DataFrame(columns=["lat", "lon", "osm_id", "landuse_type"])
        df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])
        print(f"  Found {len(df)} industrial/commercial zones.")
        return df

    def fetch_road_network(self) -> pd.DataFrame:
        """
        Fetch major road segments (motorway, trunk, primary, secondary).
        Road count per hex is used as a traffic emission proxy.

        Returns DataFrame: lat, lon, osm_id, road_type
        """
        print(f"Fetching OSM road network for bbox {self.overpass_bbox}...")
        ql = f"""
        (
          way["highway"~"motorway|trunk|primary|secondary|tertiary"]({self.overpass_bbox});
        );
        """
        data = self._query(ql)
        rows = []
        for el in data.get("elements", []):
            c = el.get("center", {})
            if not c:
                continue
            rows.append({
                "lat":       c.get("lat"),
                "lon":       c.get("lon"),
                "osm_id":    el.get("id"),
                "road_type": el.get("tags", {}).get("highway", "road"),
            })
        if not rows:
            print("  No road segments found (Overpass returned no elements).")
            return pd.DataFrame(columns=["lat", "lon", "osm_id", "road_type"])
        df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])
        print(f"  Found {len(df)} road segments.")
        return df

    def build_hex_features(self) -> pd.DataFrame:
        """
        Aggregate both industrial and road data to H3 hexes.

        Returns DataFrame with static node features:
            h3_index,
            n_industrial_zones,   - count of industrial zones in hex
            n_road_segments,      - count of major roads (traffic proxy)
            has_industry,         - binary: any industrial zone present
            road_density_score    - normalized road count (0-1)
        """
        ind_df  = self.fetch_industrial_zones()
        road_df = self.fetch_road_network()

        result_rows = {}

        # Aggregate industrial zones
        if not ind_df.empty:
            ind_df = self.mapper.add_h3_indices(ind_df)
            ind_agg = (
                ind_df.groupby("h3_index")
                .size()
                .reset_index(name="n_industrial_zones")
            )
            for _, row in ind_agg.iterrows():
                h = row["h3_index"]
                result_rows.setdefault(h, {})["n_industrial_zones"] = int(row["n_industrial_zones"])

        # Aggregate road network
        if not road_df.empty:
            road_df = self.mapper.add_h3_indices(road_df)
            road_agg = (
                road_df.groupby("h3_index")
                .size()
                .reset_index(name="n_road_segments")
            )
            for _, row in road_agg.iterrows():
                h = row["h3_index"]
                result_rows.setdefault(h, {})["n_road_segments"] = int(row["n_road_segments"])

        if not result_rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(result_rows, orient="index").reset_index()
        df = df.rename(columns={"index": "h3_index"})
        df = df.fillna(0)

        df["has_industry"] = (df.get("n_industrial_zones", 0) > 0).astype(int)

        max_roads = df.get("n_road_segments", pd.Series([1])).max()
        df["road_density_score"] = (
            df.get("n_road_segments", 0) / max(max_roads, 1)
        ).round(3)

        return df

    def save_hex_features(
        self,
        output_path: str = "data/raw/osm_features.parquet",
    ) -> pd.DataFrame:
        """Build and save hex-level OSM features."""
        df = self.build_hex_features()
        if df.empty:
            print("No OSM features to save.")
            return df
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow", index=False)
        print(f"Saved OSM hex features: {len(df)} hexes -> {output_path}")
        return df


def generate_synthetic_osm(output_path="data/raw/osm_features.parquet"):
    """
    Synthetic Pune H3-hex OSM features for demo / offline use.
    Generates realistic road-density and industrial-zone scores.
    """
    import h3
    print("Generating synthetic OSM features for Pune (demo mode)...")

    # Seed hexes: cover central Pune at resolution 8
    center_hexes = h3.h3_to_children(h3.geo_to_h3(18.52, 73.86, 7), 8)
    rng = np.random.default_rng(42)
    rows = []
    for hex_id in center_hexes:
        rows.append({
            "h3_index":           hex_id,
            "n_industrial_zones": int(rng.integers(0, 4)),
            "n_road_segments":    int(rng.integers(1, 20)),
            "has_industry":       int(rng.integers(0, 2)),
            "road_density_score": round(float(rng.random()), 3),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved {len(df)} synthetic OSM hex features -> {output_path}")
    return df


if __name__ == "__main__":
    import sys
    import time

    use_synthetic = "--synthetic" in sys.argv
    client = OSMClient(bbox="73.7,18.4,74.0,18.7", resolution=8)

    if not use_synthetic:
        print("Fetching industrial zones...")
        ind = client.fetch_industrial_zones()
        print(f"  {len(ind)} zones")

        time.sleep(3)  # Overpass rate limit courtesy delay

        print("\nFetching road network...")
        roads = client.fetch_road_network()
        print(f"  {len(roads)} road segments")

        print("\nBuilding hex features...")
        df = client.build_hex_features()
        if not df.empty:
            print(df.head())
            client.save_hex_features()
        else:
            print("  Overpass unavailable -- falling back to synthetic data.")
            generate_synthetic_osm()
    else:
        generate_synthetic_osm()

