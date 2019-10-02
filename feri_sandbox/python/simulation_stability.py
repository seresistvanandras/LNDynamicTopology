import sys, os
from analysis_utils import *

from datawand.parametrization import ParamHelper

ph = ParamHelper('../..', 'LNGraph', sys.argv)

experiment_id = ph.get("sim_res_dir")
snapshots = ph.get("snapshots")
simulation_dir = ph.get("sim_root_dir")
router_income_col = "fee"

experiment_folders = get_experiment_files(experiment_id, snapshots, simulation_dir)
source_fee = load_data(experiment_folders, snapshots, "source_fees")
router_income = load_data(experiment_folders, snapshots, "router_incomes")

corrs = ["pearson","spearman","kendall","wkendall"]

router_inc_cross = pd.DataFrame([avg_cross_corr(router_income, snap_id, router_income_col, corrs) for snap_id in snapshots])
router_traf_cross = pd.DataFrame([avg_cross_corr(router_income, snap_id, "num_trans", corrs) for snap_id in snapshots])
source_fee_cross = pd.DataFrame([avg_cross_corr(source_fee, snap_id, "mean_fee", corrs, key_col="source") for snap_id in snapshots])
source_traf_cross = pd.DataFrame([avg_cross_corr(source_fee, snap_id, "num_trans", corrs, key_col="source") for snap_id in snapshots])

router_inc_cross = reshape_cross_corr_df(router_inc_cross, corrs)
router_traf_cross = reshape_cross_corr_df(router_traf_cross, corrs)
source_fee_cross = reshape_cross_corr_df(source_fee_cross, corrs)
source_traf_cross = reshape_cross_corr_df(source_traf_cross, corrs)

router_inc_cross["statistics"] = "routing income"
router_traf_cross["statistics"] = "routing traffic"
source_fee_cross["statistics"] = "sender transaction fee"
source_traf_cross["statistics"] = "sender traffic"

stability_res = pd.concat([router_inc_cross, router_traf_cross, source_fee_cross, source_traf_cross])

stability_res = stability_res.rename({"index":"snapshot_id"}, axis=1)

output_dir = "/mnt/idms/fberes/data/bitcoin_ln_research/results/simulation_stability/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
stability_res.to_csv("%s/%s_stability_res.csv" % (output_dir,experiment_id), index=False)
print("done")