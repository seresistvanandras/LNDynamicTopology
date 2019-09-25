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
eps = ph.get("eps")
day_interval = ph.get("day_interval")
drop_disabled = ph.get("drop_disabled")
drop_low_cap = ph.get("drop_low_cap")
with_depletion = ph.get("with_depletion")

find_alternative_paths = False

data_dir = ph.get("data_dir")
#output_dir = "%s/simulations_%idays/%s/%s" % (data_dir, day_interval, snapshot_id, experiment_id)
output_dir = "%s/simulations/%s/%s" % (data_dir, snapshot_id, experiment_id)
print(output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
output_file = "%s/%s.csv" % (output_dir, experiment_id)

# 2. Load data

snapshots = pd.read_csv("%s/directed_graphs/directed_temporal_multi_edges_%idays.csv" % (data_dir, day_interval))
node_meta = pd.read_csv("%s/node_meta_with_labels.csv" % data_dir)
providers = list(node_meta["pub_key"])
edges = snapshots[snapshots["snapshot_id"]==snapshot_id]

# 3. Simulation

simulator = ts.TransactionSimulator(edges, providers, amount_sat, num_transactions, drop_disabled=drop_disabled, drop_low_cap=drop_low_cap, eps=eps, with_depletion=with_depletion)
transactions = simulator.transactions
shortest_paths, alternative_paths, all_router_fees = simulator.simulate(weight="total_fee", with_node_removals=find_alternative_paths)
total_income, total_fee = simulator.export(output_dir)

# 4. Stats

print("Total income:", total_income.sum())

# 5. Analyze optimal routing fee for nodes
if find_alternative_paths:
    opt_fees_df, p_altered = ts.calc_optimal_base_fee(shortest_paths, alternative_paths, all_router_fees)
    opt_fees_df.to_csv("%s/opt_fees.csv" % output_dir, index=False)

print("done")