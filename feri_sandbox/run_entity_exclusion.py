import pandas as pd
import os, sys, json
sys.path.insert(0,"./python")
import transaction_simulator as ts
from analysis_utils import relevant_routers

from datawand.parametrization import ParamHelper

# 1. Load parameters

ph = ParamHelper('..', 'LNGraph', sys.argv)

experiment_id = ph.get("experiment_id")
snapshot_id = ph.get("snapshot_id")
amount_sat = ph.get("amount_sat")
num_transactions = ph.get("num_transactions")
eps = ph.get("eps")
day_interval = ph.get("day_interval")

#drop_disabled = ph.get("drop_disabled")
#drop_low_cap = ph.get("drop_low_cap")
#with_depletion = ph.get("with_depletion")
#find_alternative_paths = True

drop_disabled = True
drop_low_cap = True
with_depletion = True
find_alternative_paths = False

data_dir = ph.get("data_dir")
output_dir = "%s/simulations_exclusion_%idays/%s/%s" % (data_dir, day_interval, snapshot_id, experiment_id)
print(output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Load data

node_names = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/node_names.csv")
LNBIG_nodes = list(node_names[node_names["is_lnbig"]]["pub_key"])
print(len(LNBIG_nodes))
relevant_nodes = node_names[node_names["pub_key"].isin(relevant_routers())]
print(relevant_nodes)

snapshots = pd.read_csv("%s/directed_graphs/directed_temporal_multi_edges_%idays.csv" % (data_dir, day_interval))
node_meta = pd.read_csv("%s/node_meta_with_labels.csv" % data_dir)
providers = list(node_meta["pub_key"])
edges = snapshots[snapshots["snapshot_id"]==snapshot_id]

# 3. Simulation

simulator = ts.TransactionSimulator(edges, providers, amount_sat, num_transactions, drop_disabled=drop_disabled, drop_low_cap=drop_low_cap, eps=eps, with_depletion=with_depletion, verbose=False)

global_failure_rate = {}

def get_failure_ratio():
    return len(simulator.transactions[~simulator.transactions["success"]]) / len(simulator.transactions)

# calculate original failure ratio
shortest_paths, alternative_paths, all_router_fees, _ = simulator.simulate(weight="total_fee", with_node_removals=find_alternative_paths)
global_failure_rate["all"] = get_failure_ratio()


# calculate LNBIG.com failure ratio
shortest_paths, alternative_paths, all_router_fees, _ = simulator.simulate(weight="total_fee", with_node_removals=find_alternative_paths, excluded=LNBIG_nodes)
global_failure_rate["LNBIG.com"] = get_failure_ratio()

for _, row in relevant_nodes.iterrows():
    entity_name, entity_key = row["name"], row["pub_key"]
    shortest_paths, alternative_paths, all_router_fees, _ = simulator.simulate(weight="total_fee", with_node_removals=find_alternative_paths, excluded=[entity_key])
    global_failure_rate[entity_name] = get_failure_ratio()
    
print(global_failure_rate)

with open(output_dir + "/global_failure_ratios.json", 'w') as f:
    json.dump(global_failure_rate, f)

print("done")