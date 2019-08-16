import os
import pandas as pd
from alpenglow.experiments import FactorExperiment
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

links_df = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/data/links_df_%s.csv" % str(K))

model_experiment = FactorExperiment(
    top_k=k,
    seed=seed,
    dimension=dim,
    learning_rate=0.14,
    negative_rate=neg_rate
)

rankings = model_experiment.run(links_df, exclude_known=ex_known, verbose=True)
rankings['dcg'] = DcgScore(rankings)
print(rankings["dcg"].mean())

output_dir = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/rankings/topk%i_dim%i_exk%s" % (k, dim, ex_known)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rankings.to_csv(output_dir + "/onmf_%s.csv" % str(K))

print("done")