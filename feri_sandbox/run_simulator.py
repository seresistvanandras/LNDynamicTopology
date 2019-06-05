import pandas as pd
import os, sys
import transaction_simulator as ts
from datawand.parametrization import ParamHelper

# 1. Load parameters

ph = ParamHelper('..', 'LNGraph', sys.argv)

experiment_id = ph.get("experiment_id")
snapshot_id = ph.get("snapshot_id")
amount_sat = ph.get("amount_sat")
num_transactions = ph.get("num_transactions")

data_dir = ph.get("data_dir")
output_dir = "%s/simulations/%s" % (data_dir, snapshot_id)
print(output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
output_file = "%s/%s.csv" % (output_dir, experiment_id)

# 2. Load data

snapshots = pd.read_csv("%s/directed_temporal_edges.csv" % data_dir)
node_meta = pd.read_csv("%s/node_meta_with_labels.csv" % data_dir)
providers = list(node_meta["pub_key"])
edges = snapshots[snapshots["snapshot_id"]==snapshot_id]

# 3. Simulation

simulator = ts.TransactionSimulator(edges, providers, amount_sat, num_transactions)
shortest_paths, alternative_paths, _ = simulator.simulate(weight="total_fee")
harmonic_sums, routing_differences = ts.calculate_node_influence(shortest_paths, alternative_paths)
harmonic_sums.reset_index().to_csv(output_file, index=False)

print("done")