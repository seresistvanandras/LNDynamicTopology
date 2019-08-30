import sys
from datawand.parametrization import ParamHelper

sys.path.insert(0,"../")
from link_pred_simulator import SimulatedLinkPredExperiment

ph = ParamHelper('../..', 'LinkPredSim', sys.argv)

K = ph.get("top_first_days")
top_k = ph.get("topk")
base_model = ph.get("base_model")

output_prefix = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/rankings/topk%i_exkTrue_%s/" % (top_k, str(K))

snapshot_fp = "/mnt/idms/fberes/data/bitcoin_ln_research/directed_graphs/directed_temporal_multi_edges_1days.csv"
links_fp = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/data/links_df_%s.csv" % str(K)
preds_fp = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/rankings/topk%i_exkTrue_%s/preds_%s.csv" % (top_k, str(K), base_model)
node_meta_fp = "/mnt/idms/fberes/data/bitcoin_ln_research/node_meta_with_labels.csv"

tx_fee_sat = ph.get("tx_fee_sat")
tx_num = ph.get("tx_num")
tx_alpha = ph.get("tx_alpha")
tx_drop_disabled = ph.get("tx_drop_disabled")

experiment = SimulatedLinkPredExperiment(snapshot_fp, links_fp, preds_fp, node_meta_fp, tx_fee_sat, tx_num, alpha=tx_alpha, drop_disabled=tx_drop_disabled)

experiment.preprocess()

experiment.run(".", snapshots_ids=[2], max_threads=40)
#experiment.run(output_prefix, max_threads=32)
#experiment.run(output_prefix, max_threads=16)

print("done")