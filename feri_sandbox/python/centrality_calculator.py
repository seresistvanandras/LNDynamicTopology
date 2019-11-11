import os, sys
import pandas as pd
import networkx as nx

sys.path.insert(0, "./python")
import graph_preprocessing as gp

import sys
from datawand.parametrization import ParamHelper

# Load parameters

ph = ParamHelper('../..', 'DataProcessing', sys.argv)

experiment_id = "centrality"
snapshot_id = ph.get("snapshot_id")
amount_sat = ph.get("amount_sat")
day_interval = ph.get("day_interval")
drop_disabled = ph.get("drop_disabled")
drop_low_cap = ph.get("drop_low_cap")

data_dir = ph.get("data_dir")
output_dir = "%s/simulations_%idays/%s/%s" % (data_dir, day_interval, snapshot_id, experiment_id)
print(output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Load data
    
snapshots = pd.read_csv("%s/directed_graphs/directed_temporal_multi_edges_%idays.csv" % (data_dir, day_interval))
edges = snapshots[snapshots["snapshot_id"]==snapshot_id]
filtered_edges = gp.prepare_edges_for_simulation(edges, amount_sat, drop_disabled, drop_low_cap)

# Create graph

G = nx.from_pandas_edgelist(filtered_edges, source="src", target="trg", edge_attr=["capacity", "total_fee"], create_using=nx.DiGraph())

print(G.number_of_nodes(), G.number_of_edges())

# Calculate centralities

centralities = {}
centralities["degree"] = dict(G.degree())
centralities["total_capacity"] = dict(G.degree(weight="capacity"))
centralities["total_fee"] = dict(G.degree(weight="total_fee"))
centralities["pagerank"] = dict(nx.pagerank(G))
centralities["betweeness"] = dict(nx.betweenness_centrality(G))

# Export data

centrality_df = pd.DataFrame(centralities)
centrality_df = centrality_df.reset_index().rename({"index":"node"}, axis=1)
centrality_df.to_csv("%s/scores.csv" % output_dir, index=False)