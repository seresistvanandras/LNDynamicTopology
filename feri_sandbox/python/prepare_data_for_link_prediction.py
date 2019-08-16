import sys, os
import pandas as pd
sys.path.insert(0,"..")
from ln_utils import *
from link_prediction_utils import *

from datawand.parametrization import ParamHelper
ph = ParamHelper('../..', 'LinkPrediction', sys.argv)

# Parameters
K = ph.get("top_first_days")

# Load temporal data
graph_files = []

data_dir = "../../LNdata/lncaptures/lngraph/2019/"
graph_files +=  [data_dir + f for f in sorted(os.listdir(data_dir)) if ".json" in f]
MIN_TIME = 1549065601-86400 #Saturday, February 2, 2019 12:00:01 AM
#MAX_TIME = 1552867201 #Monday, March 18, 2019 12:00:01 AM

data_dir = "../../LNdata/"
#graph_files = [data_dir + f for f in sorted(os.listdir(data_dir)) if ".json" in f]
graph_files += [data_dir + f for f in sorted(os.listdir(data_dir)) if ".json" in f][5:]
#MIN_TIME = 1552478399 # Wednesday, March 13, 2019 11:59:59 AM
MAX_TIME = 1553947199 # Saturday, March 30, 2019 11:59:59 AM

if K != None:
    graph_files = graph_files[:K]
#graph_files

EDGE_KEYS = ["node1_pub","node2_pub","last_update","capacity","channel_id",'node1_policy','node2_policy']
nodes, edges = load_temp_data(graph_files[:-1], edge_keys=EDGE_KEYS)
print(len(nodes), len(edges))

nodes = nodes[(nodes["last_update"] > MIN_TIME) & (nodes["last_update"] < MAX_TIME)]
edges = edges[(edges["last_update"] > MIN_TIME) & (edges["last_update"] < MAX_TIME)]
print(len(nodes), len(edges))

edges = edges.sort_values("last_update").reset_index(drop=True)

# Extract homophily and new channels
"""
- time of an edge channel is the 'last_update' timestamp
- we suppose: first occurrence of a channel is the creation time -> **first last_update value**
events = process_edges(edges)
"""
events = process_edges(edges)
print(events["new_channel"].value_counts())
print(events["new_edge"].value_counts())
print(events["homophily"].value_counts())
events.to_csv("/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/data/channel_events_%s.csv" % str(K), index=False)

# Prepare data for link prediction
links_df = prepare_link_prediction_data(events)
links_df.to_csv("/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/data/links_df_%s.csv" % str(K), index=False)

print("done")