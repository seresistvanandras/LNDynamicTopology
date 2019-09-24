import pandas as pd

import sys
sys.path.insert(0,"../")
from ln_utils import load_data
from transaction_simulator import get_experiment_files

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
#experiment_id = "2019-09-06_22:03:19_50000sat_k6000"
#snapshots = range(40)
#simulation_dir = "/mnt/idms/fberes/data/bitcoin_ln_research/simulations_1days/"

experiment_folders = get_experiment_files(experiment_id, snapshots, simulation_dir)
router_income = load_data(experiment_folders, snapshots, "router_incomes")
router_income_col = "fee"
all_router_incomes = pd.concat(router_income)

# LNBIG.com traffic and income timeseries
lnb_router_incomes = all_router_incomes[all_router_incomes["node"].isin(LNBIG_nodes)].groupby(["node","snapshot_id"]).agg({router_income_col:"mean","num_trans":"mean"}).sort_values(router_income_col,ascending=False).reset_index()
lnb_router_incomes = lnb_router_incomes.groupby("snapshot_id").agg({router_income_col:"sum","num_trans":"sum"}).reset_index()

#sns.lineplot(x="snapshot_id",y="num_trans",data=lnb_router_incomes)

# Calculate average routing income and traffic over snapshots and samples
all_router_incomes = all_router_incomes.groupby("node").agg({router_income_col:"mean","num_trans":"mean"}).sort_values(router_income_col,ascending=False).reset_index()
all_router_incomes = all_router_incomes.merge(node_names, left_on="node", right_on="pub_key", how="left").drop("pub_key", axis=1).set_index("node")

# nodes above 50SAT mean daily income + more than 10 transactions per day
other_routers = [
    "02ad6fb8d693dc1e4569bcedefadf5f72a931ae027dc0f0c544b34c1c6f3b9a02b",#rompert.com
    "0232e20e7b68b9b673fb25f48322b151a93186bffe4550045040673797ceca43cf",#zigzag.io
    "03e50492eab4107a773141bb419e107bda3de3d55652e6e1a41225f06a0bbf2d56",#yalls.org
    "0279c22ed7a068d10dc1a38ae66d2d6461e269226c60258c021b1ddcdfe4b00bc4",#ln1.satoshilabs.com
    "03abf6f44c355dec0d5aa155bdbdd6e0c8fefe318eff402de65c6eb2e1be55dc3e",#OpenNode
    "03c2abfa93eacec04721c019644584424aab2ba4dff3ac9bdab4e9c97007491dda",#tippin.me
    "0331f80652fb840239df8dc99205792bba2e559a05469915804c08420230e23c7c",#LightningPowerUsers.com
    "03021c5f5f57322740e4ee6936452add19dc7ea7ccf90635f95119ab82a62ae268",#bluewallet - 03021c5f5f57322740e4
    "028dcc199be86786818c8c32bffe9db8855c5fca98951eec99d1fa335d841605c2",#btc.lnetwork.tokyo
    "0217890e3aad8d35bc054f43acc00084b25229ecff0ab68debd82883ad65ee8266",#1ML.com node ALPHA
    "03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f",#ACINQ
    "02529db69fd2ebd3126fb66fafa234fc3544477a23d509fe93ed229bb0e92e4fb8",#Boltening.club
    "02cdf83ef8e45908b1092125d25c68dcec7751ca8d39f557775cd842e5bc127469",#tady je slushovo
    "03ee180e8ee07f1f9c9987d98b5d5decf6bad7d058bdd8be3ad97c8e0dd2cdc7ba",#Electrophorus [W_C_B]
    "03a503d8e30f2ff407096d235b5db63b4fcf3f89a653acb6f43d3fc492a7674019",#Sagittarius A
    "030c3f19d742ca294a55c00376b3b355c3c90d61c6b6b39554dbc7ac19b141c14f",#Bitrefill.com
    "03bb88ccc444534da7b5b64b4f7b15e1eccb18e102db0e400d4b9cfe93763aa26d",#LightningTo.Me
    "0242a4ae0c5bef18048fbecf995094b74bfb0f7391418d71ed394784373f41e4f3",#CoinGate
    "03cb7983dc247f9f81a0fa2dfa3ce1c255365f7279c8dd143e086ca333df10e278",#fairly.cheap
    "031678745383bd273b4c3dbefc8ffbf4847d85c2f62d3407c0c980430b3257c403",#lightning-roulette.com
]

node_income_and_traffic = all_router_incomes.loc[other_routers].reset_index(drop=True)
print(node_income_and_traffic)

# Export
lnbig_stats = all_router_incomes.loc[LNBIG_nodes][["fee","num_trans"]].sum()
lnbig_stats["name"] = "LNBIG.com"
node_income_and_traffic = node_income_and_traffic.append(lnbig_stats, ignore_index=True).sort_values("name")
node_income_and_traffic.to_csv("/mnt/idms/fberes/data/bitcoin_ln_research/results/router_traffic_and_income/%s.csv" % experiment_id, index=False)

print("done")