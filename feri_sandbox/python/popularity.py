import os
import pandas as pd
from alpenglow.experiments import PopularityExperiment, PopularityTimeframeExperiment
from alpenglow.evaluation import DcgScore

import sys
from datawand.parametrization import ParamHelper

ph = ParamHelper('../..', 'LinkPrediction', sys.argv)

K = ph.get("top_first_days")
k = ph.get("topk")
seed = ph.get("seed")
dim = ph.get("dim")
neg_rate = ph.get("neg_rate")
ex_known = ph.get("ex_known")

output_dir = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/rankings/topk%i_dim%i_exk%s" % (k, dim, ex_known)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

links_df = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/data/links_df_%s.csv" % str(K))

model_experiment = PopularityExperiment(
    top_k=k,
    seed=seed,
)

rankings = model_experiment.run(links_df, exclude_known=ex_known, verbose=True)
rankings['dcg'] = DcgScore(rankings)
print(rankings["dcg"].mean())

rankings.to_csv(output_dir + "/pop_%s.csv" % str(K))

print("pop done")

model_experiment = PopularityTimeframeExperiment(
    top_k=k,
    seed=seed,
)

rankings = model_experiment.run(links_df, exclude_known=ex_known, verbose=True)
rankings['dcg'] = DcgScore(rankings)
print(rankings["dcg"].mean())

rankings.to_csv(output_dir + "/time_pop_%s.csv" % str(K))

print("pop timeframe done")