import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAQClient:
    def __init__(self, bbox="73.7,18.4,74.0,18.7"):
        self.base_url = "https://api.openaq.org/v3"
        self.bbox = bbox
        
        self.api_key = os.environ.get("OPENAQ_API_KEY")
        if not self.api_key:
            print("WARNING: OPENAQ_API_KEY environment variable is not set.")
            print("OpenAQ API v3 strictly requires an API key. Please register at https://explore.openaq.org/register")
            print("Then add it to your .env file as OPENAQ_API_KEY=your_key_here")
            
        self.headers = {
            "X-API-Key": self.api_key,
            "accept": "application/json"
        }

    def fetch_data(self, days_back=30, parameter_id=2):
        """
        parameter_id=2 is typically PM2.5
        """
        if not self.api_key:
            return pd.DataFrame()
            
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"Fetching OpenAQ v3 locations in bbox {self.bbox}...")
        
        # Step 1: Find locations in the bbox that measure PM2.5
        loc_params = {
            "bbox": self.bbox,
            "parameters_id": parameter_id,
            "limit": 100
        }
        
        loc_response = requests.get(f"{self.base_url}/locations", headers=self.headers, params=loc_params)
        
        if loc_response.status_code != 200:
            print(f"Error fetching locations: {loc_response.text}")
            return pd.DataFrame()
            
        locations = loc_response.json().get("results", [])
        print(f"Found {len(locations)} locations.")
        
        all_measurements = []
        
        # Step 2: Fetch measurements for each location
        for loc in locations:
            # A location can have multiple sensors. Find the one for our parameter.
            sensors = loc.get("sensors", [])
            sensor_id = None
            for s in sensors:
                if s.get("parameter", {}).get("id") == parameter_id:
                    sensor_id = s.get("id")
                    break
                    
            if not sensor_id:
                continue
                
            print(f"Fetching measurements for Location: {loc['name']} (Sensor ID: {sensor_id})...")
            
            page = 1
            while True:
                meas_params = {
                    "date_from": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "date_to": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 1000,
                    "page": page
                }
                
                meas_url = f"{self.base_url}/sensors/{sensor_id}/measurements"
                meas_response = requests.get(meas_url, headers=self.headers, params=meas_params)
                
                if meas_response.status_code == 429:
                    print("Rate limited by OpenAQ, sleeping for 5 seconds...")
                    time.sleep(5)
                    continue
                    
                if meas_response.status_code != 200:
                    print(f"Error fetching measurements: {meas_response.text}")
                    break
                    
                data = meas_response.json().get("results", [])
                if not data:
                    break
                    
                for d in data:
                    all_measurements.append({
                        'location_id': loc['id'],
                        'location_name': loc['name'],
                        'lat': loc['coordinates']['latitude'],
                        'lon': loc['coordinates']['longitude'],
                        'parameter': 'pm25',
                        'value': d.get('value'),
                        'unit': 'µg/m³',
                        'timestamp': pd.to_datetime(d['period']['datetimeTo']['utc'])
                    })
                    
                if len(data) < 1000:
                    break
                    
                page += 1
                time.sleep(0.5)
                
        df = pd.DataFrame(all_measurements)
        return df

    def process_and_save(self, df, output_path="data/raw/openaq.parquet"):
        if df.empty:
            print("No data to save.")
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = df.drop_duplicates(subset=['location_id', 'timestamp', 'parameter'])
        
        df.to_parquet(output_path, engine="pyarrow")
        print(f"Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    # Pune bounding box
    client = OpenAQClient(bbox="73.7,18.4,74.0,18.7")
    df = client.fetch_data(days_back=7)
    client.process_and_save(df)
