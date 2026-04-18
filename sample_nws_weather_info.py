import re
import time
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt

def get_next_day_high(location_input, geolocator, session):
    # map short names to better geocoding inputs
    if location_input.strip().upper() == "MDW":
        location_input = "Chicago Midway Airport, Chicago, IL"
    if location_input.strip().upper() == "CENTRAL PARK":
        location_input = "Central Park, New York, NY"

    location = geolocator.geocode(location_input, timeout=10)
    if not location:
        return {"location": location_input, "addr": None, "temp": None, "period": None, "desc": None}

    url = f"https://forecast.weather.gov/MapClick.php?lat={location.latitude}&lon={location.longitude}"
    headers = {"User-Agent": "geo_weather_scraper_1.0 - python requests"}
    try:
        r = session.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return {"location": location_input, "addr": location.address, "temp": None, "period": None, "desc": None}

    soup = BeautifulSoup(r.content, "html.parser")
    items = soup.find_all(class_="tombstone-container")
    if not items:
        return {"location": location_input, "addr": location.address, "temp": None, "period": None, "desc": None}

    for item in items:
        period_tag = item.find(class_="period-name")
        temp_tag = item.find(class_="temp")
        short_desc_tag = item.find(class_="short-desc")
        if not period_tag or not temp_tag:
            continue
        period = period_tag.get_text(strip=True)
        temp_text = temp_tag.get_text(" ", strip=True)
        # skip today's/afternoon entries
        if "Today" in period or "Afternoon" in period:
            continue
        # only consider highs
        if "High" not in temp_text:
            continue
        # extract numeric temperature
        m = re.search(r"(-?\d+)", temp_text)
        temp_val = int(m.group(1)) if m else None
        desc = short_desc_tag.get_text(" ", strip=True) if short_desc_tag else None
        return {"location": location_input, "addr": location.address, "temp": temp_val, "period": period, "desc": desc}

    return {"location": location_input, "addr": location.address, "temp": None, "period": None, "desc": None}


if __name__ == "__main__":
    # list of requested locations
    locations = ["Central Park", "MDW", "San Francisco, CA", "Denver, CO", "Seattle, Seattle-Tacoma International Airport (KSEA)","Austin, TX", "Miami, FL", "Los Angeles, CA", "Boston, MA", "Washington DC"]

    geolocator = Nominatim(user_agent="geo_weather_scraper_1.0")
    session = requests.Session()

    results = []
    for loc in locations:
        info = get_next_day_high(loc, geolocator, session)
        results.append(info)
        time.sleep(1)  # be polite to services

    # Prepare data for horizontal bar plotting: city name (no extras) and temp only.
    filtered = [r for r in results if r.get("temp") is not None]
    if not filtered:
        print("No temperature data available to plot.")
    else:
        names = [r["location"].split(",")[0].strip() for r in filtered]
        temps = [r["temp"] for r in filtered]

        fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(names)))
        y_pos = range(len(names))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(names))]

        bars = ax.barh(y_pos, temps, color=colors, edgecolor="k")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # highest entry on top for nicer ordering

        # label each bar with the temperature value at the end of the bar
        for bar, t in zip(bars, temps):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{t}°F", va="center", ha="left", fontsize=9)

        plt.tight_layout()
        output_file = "next_day_highs_horizontal.png"
        plt.savefig(output_file, dpi=150)
        print(f"Chart saved to {output_file}")
        # try:
        #     plt.show()
        # except Exception:
        #     pass
