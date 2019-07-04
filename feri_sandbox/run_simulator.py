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
output_dir = "%s/simulations/%s/%s" % (data_dir, snapshot_id, experiment_id)
print(output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
output_file = "%s/%s.csv" % (output_dir, experiment_id)

# 2. Load data

snapshots = pd.read_csv("%s/directed_temporal_multi_edges.csv" % data_dir)
node_meta = pd.read_csv("%s/node_meta_with_labels.csv" % data_dir)
providers = list(node_meta["pub_key"])
edges = snapshots[snapshots["snapshot_id"]==snapshot_id]

# 3. Simulation

simulator = ts.TransactionSimulator(edges, providers, amount_sat, num_transactions)
transactions = simulator.transactions
shortest_paths, alternative_paths, all_router_fees = simulator.simulate(weight="total_fee")
total_income, total_fee = simulator.export(output_dir)
#harmonic_sums, routing_differences = ts.calculate_node_influence(shortest_paths, alternative_paths)
#harmonic_sums.reset_index().to_csv(output_file, index=False)

# 4. Stats

print("Total income:", total_income.sum())

# 5. Analyze optimal routing fee for nodes
opt_fees_df, p_altered = ts.calc_optimal_base_fee(shortest_paths, alternative_paths, all_router_fees)
opt_fees_df.to_csv("%s/opt_fees.csv" % output_dir, index=False)
print("done")