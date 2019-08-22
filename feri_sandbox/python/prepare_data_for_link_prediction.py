import sys, os
import pandas as pd
sys.path.insert(0,"..")
from ln_utils import *
from link_prediction_utils import *

from datawand.parametrization import ParamHelper
ph = ParamHelper('../..', 'LinkPrediction', sys.argv)

# Parameters
K = ph.get("top_first_days")

edges = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/directed_graphs/directed_temporal_multi_edges_1days.csv")

if K != None:
    edges = edges[edges["snapshot_id"]<K]

edges = edges.sort_values(["snapshot_id","last_update"]).reset_index(drop=True)

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