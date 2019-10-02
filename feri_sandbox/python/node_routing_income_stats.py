import pandas as pd
import numpy as np
import os

import sys
sys.path.insert(0,"../")
from analysis_utils import *

from datawand.parametrization import ParamHelper
ph = ParamHelper('../..', 'LNGraph', sys.argv)

node_names = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/node_names.csv")
print(node_names.head())

LNBIG_nodes = list(node_names[node_names["is_lnbig"]]["pub_key"])
print(len(LNBIG_nodes))

node_names = node_names[["name","pub_key"]]

experiment_id = ph.get("sim_res_dir")
snapshots = ph.get("snapshots")
simulation_dir = ph.get("sim_root_dir")
#experiment_id = "60000sat_k7000_eps0.80"
#snapshots = range(40)
#simulation_dir = "/mnt/idms/fberes/data/bitcoin_ln_research/simulations_multi/"

experiment_folders = get_experiment_files(experiment_id, snapshots, simulation_dir)
router_income = load_data(experiment_folders, snapshots, "router_incomes")
all_router_incomes = pd.concat(router_income)

# LNBIG.com traffic and income timeseries
#lnb_router_incomes = all_router_incomes[all_router_incomes["node"].isin(LNBIG_nodes)].groupby(["node","snapshot_id"])[["fee","num_trans"]].mean().sort_values("fee", ascending=False).reset_index()
#lnb_router_incomes = lnb_router_incomes.groupby("snapshot_id")[["fee","num_trans"]].sum().reset_index()
#sns.lineplot(x="snapshot_id",y="num_trans",data=lnb_router_incomes)

max_sample_id = all_router_incomes["sample"].max()
num_simulations = len(snapshots) * max_sample_id
print(max_sample_id)

all_router_incomes = all_router_incomes.groupby("node")[["fee","num_trans"]].sum() / num_simulations
all_router_incomes = all_router_incomes.rename({"fee":"mean_fee", "num_trans":"mean_num_trans"}, axis=1)
all_router_incomes = all_router_incomes.reset_index()

# Calculate average routing income and traffic over snapshots and samples for every node
#groups = all_router_incomes.groupby("node")
#aggregated = groups.agg({
#    "fee" : {"mean_fee":np.mean,"std_fee":np.std},
#    "num_trans" : {"mean_num_trans":np.mean,"std_num_trans":np.std},
#})
#aggregated.columns = aggregated.columns.droplevel(0)
#all_router_incomes = aggregated.reset_index().sort_values(["mean_fee","mean_num_trans"], ascending=False)

all_router_incomes = all_router_incomes.merge(node_names, left_on="node", right_on="pub_key", how="left").drop("pub_key", axis=1).set_index("node")

#all_router_incomes.to_csv("mean_node_stats.csv")
#all_router_incomes[all_router_incomes["num_trans"]>=10.0].to_csv("mean_node_stats_10.csv")
#print(all_router_incomes[all_router_incomes["mean_num_trans"]>=10.0].head(50))

other_routers = relevant_routers()

selected_node_stats = all_router_incomes.loc[other_routers].reset_index(drop=True)

# Export
lnbig_stats = all_router_incomes.loc[LNBIG_nodes][["mean_fee","mean_num_trans"]].sum()
lnbig_stats["name"] = "LNBIG.com"
selected_node_stats = selected_node_stats.append(lnbig_stats, ignore_index=True).sort_values("mean_fee", ascending=False)

print(selected_node_stats)

output_dir = "/mnt/idms/fberes/data/bitcoin_ln_research/results/router_traffic_and_income"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

selected_node_stats.to_csv("%s/%s_selected.csv" % (output_dir, experiment_id), index=False)
all_router_incomes.to_csv("%s/%s_all.csv" % (output_dir, experiment_id), index=True)

print("done")