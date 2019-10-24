import pandas as pd
import os, sys
sys.path.insert(0,"./python")
import transaction_simulator as ts
from datawand.parametrization import ParamHelper

# 1. Load parameters

ph = ParamHelper('..', 'LNGraph', sys.argv)

data_dir = ph.get("data_dir")
amount_sat = ph.get("amount_sat")
num_transactions = ph.get("num_transactions")
eps = ph.get("eps")
drop_disabled = True
drop_low_cap = True
with_depletion = True
find_alternative_paths = False

snapshots = list(range(40))
samples = list(range(10))

all_snapshots = pd.read_csv("%s/directed_graphs/directed_temporal_multi_edges_1days.csv" % data_dir)
node_meta = pd.read_csv("%s/node_meta_with_labels.csv" % data_dir)
providers = list(node_meta["pub_key"])

CNT = 0
for snapshot_id in snapshots:
    for sample_id in samples:
        experiment_id = "%isat_k%i_eps%.2f.%i" % (amount_sat, num_transactions, eps, sample_id)
        output_dir = "%s/simulations_param_tuning/%s/%s" % (data_dir, snapshot_id, experiment_id)
        print(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Load data

        snapshots = all_snapshots.copy()
        edges = snapshots[snapshots["snapshot_id"]==snapshot_id]

        # 3. Simulation
        
        simulator = ts.TransactionSimulator(edges, providers, amount_sat, num_transactions, drop_disabled=drop_disabled, drop_low_cap=drop_low_cap, eps=eps, with_depletion=with_depletion)
        transactions = simulator.transactions
        shortest_paths, alternative_paths, all_router_fees, _ = simulator.simulate(weight="total_fee", with_node_removals=find_alternative_paths)
        total_income, total_fee = simulator.export(output_dir)
        
        print()
        print("done:", CNT)
        print()
        CNT += 1
print("DONE")