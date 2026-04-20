import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_dummy_data(filename="data/traffic.csv", rows=1000):
    start_date = datetime(2023, 1, 1, 0, 0)
    
    dates = [start_date + timedelta(hours=i) for i in range(rows)]
    base_volume = 500
    
    # Adding some seasonality and noise to traffic volume
    volumes = []
    for dt in dates:
        # Higher traffic during day, lower at night
        hour_effect = np.sin(np.pi * (dt.hour - 6) / 12) if 6 <= dt.hour <= 18 else -0.5
        # Lower on weekends
        weekend_effect = -100 if dt.weekday() >= 5 else 0
        
        noise = np.random.normal(0, 50)
        volume = max(0, int(base_volume + hour_effect * 300 + weekend_effect + noise))
        volumes.append(volume)

    data = {
        "Date": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dates],
        "Area Name": ["Central"] * rows,
        "Road/Intersection Name": ["Main St"] * rows,
        "Traffic Volume": volumes,
        "Average Speed": np.random.uniform(20, 60, rows).round(2)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Dummy data generated at {filename} with {rows} rows.")

if __name__ == "__main__":
    create_dummy_data()
