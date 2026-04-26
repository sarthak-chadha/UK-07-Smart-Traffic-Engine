# DeepLearn Navigation Engine: Architecture & Implementation Guide

This document provides a comprehensive, step-by-step walkthrough of the DeepLearn project—a Graph Neural Network (GNN)-powered smart routing engine. It covers the entire lifecycle from raw dataset creation to the completed web application, detailing the complex parts of the codebase.

## Step 1: Downloading the Base Road Network (`project_Step_1.py`)
The project starts by defining a geographic bounding box spanning Dehradun and Mussoorie. 
Using `osmnx`, the code fetches the full road network (nodes and edges) from OpenStreetMap.
It filters for drivable roads (rejecting private pathways or footpaths) and calculates base edge speeds and travel times based on the road typologies. 
The base network is stored locally as an XML-based GraphML format in `Dataset/dehradun_mussoorie_full.graphml`.

## Step 2: Feature Engineering & Topography (`project_Step_2.py`)
Mountainous terrain like Mussoorie requires 3D topologies, so this step augments the base map.
It fetches elevation data from the `opentopodata` API (SRTM 30m) and assigns gradient/grade values to each road segment. 
Features are extracted and standardized using Scikit-Learn's `StandardScaler` to ensure numerical stability during neural network training. 

**Complex Concept Explained: Creating the PyTorch Geometric Graph Tensor**
```python
graph_data = Data(
    x          = node_x,         # [num_nodes, 3 or 4]
    edge_index = edge_index_t,   # [2, num_edges]
    edge_attr  = edge_attr,      # [num_edges, 15]
)
```
*Detailing the code block*: 
Instead of a simple Pandas dataframe, the map data is shaped into a graph structural format required by PyTorch Geometric. 
- `x` holds standardized **Node Features** (latitude, longitude, degree, and elevation).
- `edge_index` acts as the adjacency matrix describing the connectivity array (which source node index connects to which destination node index).
- `edge_attr` stores the **Edge Features** (length, base speed, travel time, one-way flag, lanes, grade, absolute grade, and an 8-column one-hot encoded representation of highway types like motorway, trunk, residential, etc). All Continuous indices are strictly normalized.

## Step 3: Fetching Live Traffic Labels (`project_Step_3.py`)
This script executes sequential pulls from the TomTom Traffic API to generate live historical traffic snapshots. It processes edge midpoints to locate accurate road segments. 

*Detailing the TomTom API logic*: It queries the API to determine the "freeFlowSpeed" versus the "currentSpeed" to calculate a congestion traffic ratio (e.g., if cars move at 15km/h on a 30km/h road, the factor is `0.5`). To prevent exhausting API quotas, segments are cached, and rate-limiting (HTTP 429) exponential backoffs are implemented.

## Step 4: GNN Training & The ElevationAwareGAT Engine (`project_Step_4.py`)
This is the Machine Learning center of the framework. It consolidates previous time-snapshots, merges temporal periodicity (sin/cos representation of time of day), and processes lag features. There are two highly complex components here.

### 1. The GATv2 Architecture
```python
class ElevationAwareGAT(torch.nn.Module):
    def __init__(self, ...):
        # ...
        self.conv1 = GATv2Conv(...)
        self.res_proj = torch.nn.Linear(node_in, hidden_dim * heads, bias=False)
        # ...
        
    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.bn1(h)
        # Residual add logic to prevent over-smoothing
        h = F.elu(h + self.res_proj(x))
        # ...
```
*Detailing the code block*: `GATv2Conv` (Graph Attention Network v2) allows the model to dynamically pay "attention" to adjacent nodes depending on edge features. A critical architectural addition here is the **residual connection (`self.res_proj(x)`)**. When Graph Neural Networks perform message passing across multiple layers, node features often blur together mathematically (a flaw known as over-smoothing). By injecting a linear projection of the original unmodified `x` tensor—which holds exact physical topography data like `elevation` and `coordinates`—the model retains critical raw geography instead of blurring the mountainous terrain into a flat feature-average. The attention module's final operation is a fully connected MLP edge regressor that yields a traffic factor prediction ranging from `0` to `1`.

### 2. Custom Weighted MSE Loss
```python
def weighted_mse(pred, target, confidence, highway_weight):
    combined   = confidence * highway_weight
    is_slight  = ((target < 0.97) & (target >= 0.85)).float()
    is_heavy   = (target < 0.85).float()
    cong_fac   = (1.0 + 29.0 * is_slight + 119.0 * is_heavy)
    return (combined * cong_fac * (pred - target) ** 2).mean()
```
*Detailing the code block*: In traffic modeling, ~99% of normal roads are completely free-flowing. A standard Loss Function would just constantly predict "Free Flow", achieving 99% global accuracy while failing completely at its only true purpose (predicting traffic jams). This custom Mean Squared Error selectively and aggressively amplifies errors during congestion: Heavy traffic errors are multiplied by 120x (`HEAVY_UPWEIGHT`) and slight traffic by 30x. It further multiplies mathematically by `confidence` (TomTom API accuracy validation) and `highway_weight` to ensure main arterial roads punish the model more than minor dirt trails, effectively balancing an unbalanced dataset dynamically.

## Building the Interface: `app.py` and `TerrainNav_App`

### The FastAPI Backend Router (`app.py`)
The Python server bridges the Deep Learning model and the traditional pathfinding algorithm together.
When the user submits a route:
1. **Geocoding**: Converts text addresses entered by the user (e.g. "Clock Tower") to coordinates using a constrained TomTom API search geographically locked to the center of Dehradun explicitly (`radius=45000`).
2. **GNN Edge Inference**: Generates the periodicity temporal features for "Right Now" and runs the entire road map through `traffic_model_v1.pth` dynamically to estimate traffic delay modifiers over every road on the grid.
3. **Weight Adjustments & Dijkstra**: Performs `nx.set_edge_attributes(G, dict(zip(G.edges(keys=True), dynamic_weights)), 'weight')` where new spatial weight is calculated as exact length / modified dynamic speed (`dynamic_weights = edge_lengths / (speed_mps * factors_clipped)`). The algorithm `nx.shortest_path()` then traces a Dijkstra shortest route using these localized 'weight' times.

### The Interactive App Frontend (`TerrainNav_App/`)
A Progressive Web App (PWA) built specifically with Vanilla JavaScript, HTML, and CSS (managing mapping tiles via an integration tool like Leaflet.js fetching OSM). 
It features a `manifest.json` and a Service Worker (`sw.js`). This layout is utilized so the framework can be installed natively as an 'App' directly onto mobile device homescreens without going through an App Store. 
- `api.js` connects cross-origin to the backend predictions.
- `app.js` runs event listeners to capture exact Android/iOS GPS coordinates for instantly setting starting lines.
- `map.js` parses GeoJSON/coordinates emitted by `app.py` and paints the optimal route visually across dynamic tiles.
