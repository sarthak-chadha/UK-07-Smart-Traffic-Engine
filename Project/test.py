import torch
import glob
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Dataset_DIR = os.path.join(BASE_DIR, "Dataset")

def check_traffic_distribution():
    files = glob.glob(os.path.join(Dataset_DIR, "dm_graph_live_*.pt"))
    all_y = []

    for f in files:
        data = torch.load(f, weights_only=False)
        all_y.append(data.y.view(-1))

    if not all_y:
        print("No snapshots found!")
        return

    full_y = torch.cat(all_y)
    
    # Calculate stats
    total_edges = full_y.numel()
    jammed_edges = (full_y < 0.7).sum().item()
    jam_percentage = (jammed_edges / total_edges) * 100

    print("=" * 40)
    print("DATASET TRAFFIC ANALYSIS")
    print("=" * 40)
    print(f"Total Snapshots : {len(files)}")
    print(f"Total Edges     : {total_edges:,}")
    print(f"Min Factor      : {full_y.min().item():.4f}")
    print(f"Max Factor      : {full_y.max().item():.4f}")
    print(f"Mean Factor     : {full_y.mean().item():.4f}")
    print("-" * 40)
    print(f"Congested Edges (< 0.7): {jammed_edges:,}")
    print(f"Congestion Density     : {jam_percentage:.2f}%")
    print("=" * 40)

    if jam_percentage < 5:
        print("WARNING: Low congestion density detected (< 5%). Dataset may be skewed towards free-flow conditions.")

if __name__ == "__main__":
    check_traffic_distribution()