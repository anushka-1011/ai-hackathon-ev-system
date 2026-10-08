import requests
import pandas as pd

START_DATE = "2019-04-01"
END_DATE = "2019-05-31"

sites = {
    "Caltech": (34.13659, -118.12721),
    "JPL": (34.20000, -118.17167),
    "Office001": (37.37000, -122.04000)
}

for site, (lat, lon) in sites.items():

    print(f"\nDownloading weather for {site}...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "weather_code"
        ),
        "timezone": "America/Los_Angeles"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    weather = pd.DataFrame(data["hourly"])

    weather["site_name"] = site

    filename = f"{site.lower()}_weather.csv"
    weather.to_csv(filename, index=False)

    print(f"Saved: {filename}")
    print(f"Rows: {len(weather)}")

print("\n WEATHER DOWNLOAD COMPLETE!")