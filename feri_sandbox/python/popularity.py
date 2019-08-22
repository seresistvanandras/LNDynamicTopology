import os, sys
import pandas as pd
from alpenglow.experiments import PopularityExperiment, PopularityTimeframeExperiment 
from datawand.parametrization import ParamHelper

sys.path.insert(0,"../")
from link_prediction_utils import get_rankings

ph = ParamHelper('../..', 'LinkPrediction', sys.argv)

K = ph.get("top_first_days")
k = ph.get("topk")
seed = ph.get("seed")
ex_known = ph.get("ex_known")

output_dir = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/rankings/topk%i_exk%s_%s" % (k, ex_known, str(K))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

links_df = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/data/links_df_%s.csv" % str(K))

model_experiment = PopularityExperiment(
    top_k=k,
    seed=seed,
)

pred_file = output_dir + "/preds_pop.csv"
rankings = get_rankings(model_experiment, links_df, ex_known, pred_file=pred_file)
rankings.to_csv(output_dir + "/pop.csv")

print("pop done")

model_experiment = PopularityTimeframeExperiment(
    top_k=k,
    seed=seed,
)

pred_file = output_dir + "/preds_time_pop.csv"
rankings = get_rankings(model_experiment, links_df, ex_known, pred_file=pred_file)
rankings.to_csv(output_dir + "/time_pop.csv")

print("pop timeframe done")