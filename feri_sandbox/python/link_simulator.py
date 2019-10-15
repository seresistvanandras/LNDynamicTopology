import sys
from datawand.parametrization import ParamHelper

from link_sim_experiment import SimulatedLinkPredExperiment

ph = ParamHelper('../..', 'LinkPredSim', sys.argv)

K = 20#ph.get("top_first_days")
top_k = 500#ph.get("topk")

output_prefix = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/rankings/topk%i_exkTrue_%s/" % (top_k, str(K))

snapshot_fp = "/mnt/idms/fberes/data/bitcoin_ln_research/directed_graphs/directed_temporal_multi_edges_1days.csv"
links_fp = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/data/links_df_%s.csv" % str(K)
node_meta_fp = "/mnt/idms/fberes/data/bitcoin_ln_research/node_meta_with_labels.csv"

tx_fee_sat = ph.get("tx_fee_sat")
tx_num = ph.get("tx_num")
epsilon = ph.get("epsilon")
drop_disabled = ph.get("drop_disabled")
drop_low_cap = ph.get("drop_low_cap")
with_depletion = ph.get("with_depletion")
time_window = ph.get("time_window")
half_life = ph.get("half_life")

experiment = SimulatedLinkPredExperiment(top_k, half_life, snapshot_fp, links_fp, node_meta_fp, tx_fee_sat, tx_num, epsilon, drop_disabled, drop_low_cap, with_depletion, time_window, only_triangles=False, verbose=False)

experiment.preprocess()

#experiment.run(".", snapshots_ids=[2], max_threads=20)
experiment.run(output_prefix, max_threads=1)
#experiment.run(output_prefix, max_threads=16)

print("done")