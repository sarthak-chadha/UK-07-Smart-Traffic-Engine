import osmnx as ox
import torch
import re
import pickle
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ox.settings.log_console = True
ox.settings.use_cache   = True

print("=" * 40)
print("STEP 2: Feature Engineering + Elevation")
print("=" * 40)
data_path = os.path.join(BASE_DIR, "Dataset", "dehradun_mussoorie_full.graphml")
if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found. Run Step 1 first.")
    exit()

G = ox.load_graphml(data_path)
print(f"Loaded: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")

ELEVATION_URL = "https://api.opentopodata.org/v1/srtm30m?locations={locations}"
print("Fetching elevation data... (may take 2-5 mins for large graphs)")
try:
    ox.settings.elevation_url_template = ELEVATION_URL
    G = ox.elevation.add_node_elevations_google(
        G,
        api_key=None,   # key=None → uses ox.settings.elevation_url_template
        batch_size=100,
        pause=1,
    )
    G = ox.elevation.add_edge_grades(G, add_absolute=True)
    elevation_ok = True
except Exception as e:
    print(f"WARNING: Elevation fetch failed ({e}). Continuing without it.")
    elevation_ok = False

G = ox.project_graph(G)
nodes_df, edges_df = ox.graph_to_gdfs(G)

if elevation_ok:
    print(f"Elevation OK: {nodes_df['elevation'].min():.0f}m – {nodes_df['elevation'].max():.0f}m")


def first_val(val):
    """OSM sometimes stores tags as Python lists — always return the first item."""
    if isinstance(val, list):
        return val[0]
    s = str(val).strip()
    if s.startswith("["):
        try:
            parsed = eval(s)
            if isinstance(parsed, list):
                return str(parsed[0])
        except Exception:
            pass
    return val

def clean_numeric(val, default=1.0):
    val = first_val(val)
    match = re.search(r"(\d+\.?\d*)", str(val))
    return float(match.group(1)) if match else default

def clean_highway(val):
    h = str(first_val(val)).lower().strip()
    return h.replace("_link", "")

def clean_speed(val, default=30.0):
    """Parse speed_kph, converting mph → kph when needed."""
    val = first_val(val)
    s   = str(val).lower()
    num = re.search(r"(\d+\.?\d*)", s)
    if not num:
        return default
    speed = float(num.group(1))
    if "mph" in s:
        speed *= 1.60934
    return speed

node_list = list(G.nodes())
mapping   = {node: i for i, node in enumerate(node_list)}

node_feats = nodes_df[["y", "x"]].copy()   # y=lat, x=lon (from WGS84 before projection; preserved by osmnx)
node_feats["degree"] = pd.Series(dict(G.degree()))

if elevation_ok:
    node_feats["elevation"] = nodes_df["elevation"]
    feat_names = "[lat, lon, degree, elevation]"
else:
    feat_names = "[lat, lon, degree]"

node_scaler = StandardScaler()
node_x = torch.tensor(
    node_scaler.fit_transform(node_feats.fillna(0).values),
    dtype=torch.float
)
print(f"\nNode features  : {node_x.shape}  {feat_names}")

CATEGORIES = [
    "motorway", "trunk", "primary", "secondary",
    "tertiary", "residential", "service", "track"
]

edge_rows  = []
edge_index = []

for u, v, k, data in G.edges(keys=True, data=True):
    edge_index.append([mapping[u], mapping[v]])

    length      = float(data.get("length", 0.0))
    speed       = clean_speed(data.get("speed_kph", 30.0))
    travel_time = float(data.get("travel_time",
                        length / max(speed * 1000 / 3600, 1e-6)))
    oneway      = 1.0 if str(data.get("oneway", "False")).lower() == "true" else 0.0
    lanes       = clean_numeric(data.get("lanes", 1.0))

    if elevation_ok:
        grade     = float(data.get("grade",     0.0))
        grade_abs = float(data.get("grade_abs", abs(grade)))
    else:
        grade, grade_abs = 0.0, 0.0

    hwy     = clean_highway(data.get("highway", "unclassified"))
    hwy_vec = [1.0 if cat == hwy else 0.0 for cat in CATEGORIES]
    if sum(hwy_vec) == 0:
        hwy_vec[-1] = 1.0   # unrecognised type → 'track/other' bucket

    edge_rows.append([length, speed, travel_time, oneway, lanes, grade, grade_abs] + hwy_vec)

edge_feat_matrix = np.array(edge_rows, dtype=np.float32)

CONT_IDX = [0, 1, 2, 5, 6]
edge_scaler = StandardScaler()
edge_feat_matrix[:, CONT_IDX] = edge_scaler.fit_transform(
    edge_feat_matrix[:, CONT_IDX]
)

edge_attr    = torch.tensor(edge_feat_matrix, dtype=torch.float)
edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

print(f"Edge features  : {edge_attr.shape}  "
      f"[length, speed, travel_time, oneway, lanes, grade, grade_abs, + 8 one-hot]")

graph_data = Data(
    x          = node_x,         # [num_nodes, 3 or 4]
    edge_index = edge_index_t,   # [2, num_edges]
    edge_attr  = edge_attr,      # [num_edges, 15]
)

assert graph_data.num_nodes == len(node_list), \
    f"Node mismatch: {graph_data.num_nodes} vs {len(node_list)}"
assert graph_data.num_edges == len(edge_rows), \
    f"Edge mismatch: {graph_data.num_edges} vs {len(edge_rows)}"

print(f"Validation     : PASSED ✓")

dataset_dir = os.path.join(BASE_DIR, "Dataset")
models_dir  = os.path.join(BASE_DIR, "Models")
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(models_dir,  exist_ok=True)

tensor_path = os.path.join(dataset_dir, "dm_graph_tensors.pt")
torch.save(graph_data, tensor_path)

scaler_path = os.path.join(models_dir, "scalers.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(
        {
            "node_scaler"        : node_scaler,
            "edge_scaler"        : edge_scaler,
            "node_feature_names" : list(node_feats.columns),
            "edge_cont_indices"  : CONT_IDX,
        },
        f,
    )

print("\n" + "=" * 40)
print("STEP 2 COMPLETE")
print(f"  Nodes          : {graph_data.num_nodes}")
print(f"  Edges          : {graph_data.num_edges}")
print(f"  Node feat dim  : {graph_data.num_node_features}")
print(f"  Edge feat dim  : {graph_data.num_edge_features}")
if elevation_ok:
    print(f"  Elevation      : {node_feats['elevation'].min():.0f}m – {node_feats['elevation'].max():.0f}m")
print(f"  Graph tensor   : {tensor_path}")
print(f"  Scalers        : {scaler_path}")
print("=" * 40)