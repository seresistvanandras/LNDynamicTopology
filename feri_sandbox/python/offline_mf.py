import os, sys
import pandas as pd
from alpenglow.experiments import BatchFactorExperiment
from datawand.parametrization import ParamHelper

sys.path.insert(0,"../")
from link_prediction_utils import get_rankings

ph = ParamHelper('../..', 'LinkPrediction', sys.argv)

K = ph.get("top_first_days")
k = ph.get("topk")
seed = ph.get("seed")
dim = ph.get("dim")
neg_rate = ph.get("neg_rate")
l_rate = 0.05
ex_known = ph.get("ex_known")

output_dir = "/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/rankings/topk%i_exk%s_%s" % (k, ex_known, str(K))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

links_df = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/link_prediction/data/links_df_%s.csv" % str(K))

model_experiment = BatchFactorExperiment(
    top_k=k,
    seed=seed,
    dimension=dim,
    learning_rate=l_rate,
    negative_rate=neg_rate
)

model_id = "offmf_dim%i_lr%0.3f_nr%i" % (dim, l_rate, neg_rate)
pred_file = None#output_dir + "/preds_%s.csv" % model_id
rankings = get_rankings(model_experiment, links_df, ex_known, pred_file=pred_file)
rankings.to_csv(output_dir + "/%s.csv" % model_id)