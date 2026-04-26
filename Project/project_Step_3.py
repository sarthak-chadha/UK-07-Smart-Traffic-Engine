import torch
import requests
import osmnx as ox
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Each time slot uses its own API key to keep each one under 2,500 calls/day.
# .env should have: TOMTOM_KEY_MORNING, TOMTOM_KEY_AFTERNOON, TOMTOM_KEY_EVENING
SLOTS = {
    "1": ("morning",   "TOMTOM_KEY_MORNING",   "8:30–9:00 AM"),
    "2": ("afternoon", "TOMTOM_KEY_AFTERNOON",  "2:00–3:00 PM"),
    "3": ("evening",   "TOMTOM_KEY_EVENING",    "5:30–6:30 PM"),
}

print("Select time slot:")
for num, (name, _, time_range) in SLOTS.items():
    print(f"  {num}. {name.capitalize()}  ({time_range})")

choice = input("Enter 1 / 2 / 3: ").strip()
if choice not in SLOTS:
    print("Invalid choice. Enter 1, 2, or 3.")
    sys.exit(1)

SLOT_NAME, ENV_VAR, SLOT_TIME = SLOTS[choice]
API_KEY = os.getenv(ENV_VAR)

if not API_KEY:
    print(f"ERROR: {ENV_VAR} not found in .env file!")
    sys.exit(1)

print(f"Using {SLOT_NAME} key  [{SLOT_TIME}]")

TENSOR_PATH = os.path.join(BASE_DIR, "Dataset", "dm_graph_tensors.pt")
MAP_PATH    = os.path.join(BASE_DIR, "Dataset", "dehradun_mussoorie_full.graphml")

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_PATH   = os.path.join(BASE_DIR, "Dataset", f"dm_graph_live_{RUN_TIMESTAMP}.pt")

LIVE_HIGHWAY_TYPES = {
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'trunk', 'trunk_link', 'motorway', 'motorway_link'
}

# TomTom Free Tier: 5 req/second max
SLEEP_INTERVAL = 0.22
MAX_RETRIES    = 3

# Set DRY_RUN = True to count API calls WITHOUT spending any quota.
# Runs in seconds — use this first to know exactly how many credits one run costs.
DRY_RUN = False

print("=" * 40)
if DRY_RUN:
    print("STEP 3: DRY RUN (no API calls)")
else:
    print("STEP 3: Sequential Live Traffic Fetch")
print("=" * 40)

for path in [TENSOR_PATH, MAP_PATH]:
    if not os.path.exists(path):
        print(f"ERROR: Missing {path}. Run previous steps first.")
        sys.exit(1)

graph_data = torch.load(TENSOR_PATH, weights_only=False)
G = ox.load_graphml(MAP_PATH)

traffic_y = []
conf_list = []
api_cache = {} # Cache to avoid paying for the same road segment twice
stats = {"api_calls": 0, "cache_hits": 0, "hikes": 0}

edges = list(G.edges(keys=True, data=True))
print(f"Processing {len(edges)} edges...")

with requests.Session() as session:
    for u, v, k, data in tqdm(edges, desc="Fetching Traffic"):
        hwy = str(data.get('highway', '')).lower().strip("[]' \"")

        if any(m in hwy for m in LIVE_HIGHWAY_TYPES):
            # Geometry midpoint is more accurate for winding mountain roads.
            # G is WGS84 so midpoint.y/x are valid lat/lon directly.
            if 'geometry' in data:
                midpoint = data['geometry'].interpolate(0.5, normalized=True)
                lat, lon = round(midpoint.y, 4), round(midpoint.x, 4)
            else:
                lat = round((G.nodes[u]['y'] + G.nodes[v]['y']) / 2, 4)
                lon = round((G.nodes[u]['x'] + G.nodes[v]['x']) / 2, 4)

            cache_key = (lat, lon)

            if cache_key in api_cache:
                factor, confidence = api_cache[cache_key]
                stats["cache_hits"] += 1
            else:
                if DRY_RUN:
                    # Just count — don't call the API
                    factor, confidence = 1.0, 0.0
                else:
                    url = (
                        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
                        f"absolute/10/json?key={API_KEY}&point={lat},{lon}"
                    )
                    factor, confidence = 1.0, 0.0  # safe defaults (clear road, zero confidence)
                    for attempt in range(MAX_RETRIES):
                        try:
                            resp = session.get(url, timeout=10)
                            if resp.status_code == 200:
                                flow = resp.json().get('flowSegmentData', {})
                                curr = float(flow.get('currentSpeed', 30))
                                free = float(flow.get('freeFlowSpeed', 30))
                                confidence = float(flow.get('confidence', 1.0))
                                factor = min(curr / max(free, 1.0), 1.0)
                                break
                            elif resp.status_code == 429:  # rate limited — back off exponentially
                                time.sleep(2 ** (attempt + 1))
                        except Exception:
                            time.sleep(1.5 * (attempt + 1))
                    time.sleep(SLEEP_INTERVAL)

                api_cache[cache_key] = (factor, confidence)
                stats["api_calls"] += 1
        else:
            factor, confidence = 1.0, 1.0  # small streets / hike trails: clear
            stats["hikes"] += 1

        traffic_y.append(factor)
        conf_list.append(confidence)

if len(traffic_y) != graph_data.num_edges:
    print(f"ERROR: Got {len(traffic_y)} labels for {graph_data.num_edges} edges. Cannot save.")
    sys.exit(1)

graph_data.y          = torch.tensor(traffic_y, dtype=torch.float).view(-1, 1)
graph_data.confidence = torch.tensor(conf_list, dtype=torch.float).view(-1, 1)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save(graph_data, OUTPUT_PATH)

print("\n" + "=" * 40)
print("SUCCESS")
print(f"  Snapshot    : {RUN_TIMESTAMP}")
print(f"  API Calls   : {stats['api_calls']}")
print(f"  Cache Hits  : {stats['cache_hits']}")
print(f"  Saved File  : {OUTPUT_PATH}")
print("=" * 40)