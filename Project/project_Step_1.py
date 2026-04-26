import osmnx as ox
import networkx as nx
import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 600

target_folder = os.path.join(BASE_DIR, "Dataset")
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

cache_dir = "cache"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cleared old cache.")



north = 30.60   # well above Mussoorie
south = 30.18   # well below Dehradun
east  = 78.25   # east edge of both cities
west  = 77.78   # west edge of Dehradun


road_filter = (
    '["highway"~"motorway|motorway_link|trunk|trunk_link'
    '|primary|primary_link|secondary|secondary_link'
    '|tertiary|tertiary_link|unclassified|residential'
    '|living_street|service|track|road"]'
    '["access"!~"no|private"]'
    '["motor_vehicle"!~"no"]'
)

print("Downloading full Dehradun + Mussoorie road network...")
print(f"  Bounding box: N={north}, S={south}, E={east}, W={west}")

G = ox.graph_from_bbox(bbox=(west, south, east, north), custom_filter=road_filter)

G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
save_path = os.path.join(target_folder, "dehradun_mussoorie_full.graphml")
ox.save_graphml(G, filepath=save_path)

print("-" * 30)
print("SUCCESS!")
print(f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)}")
print(f"Saved to: {save_path}  (WGS84 — projection done in Step 2)")
print("-" * 30)

ox.plot_graph(G, node_size=0, edge_color="lime", edge_linewidth=0.2)