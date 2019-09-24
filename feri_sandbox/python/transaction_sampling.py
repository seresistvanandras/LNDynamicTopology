import pandas as pd
import numpy as np

def get_src_proba(df, alpha):
    df["src_proba"] = alpha / (alpha + df["degree"])
    df["src_proba"] = df["src_proba"] / df["src_proba"].sum()
    
def get_src_rayleigh_proba(df):
    s = df["total_capacity"].median()
    print("total capacity median: %i" % s)
    ss = s**2
    x = np.array(df["total_capacity"], dtype="float64")
    df["src_proba"] = (x/ss) * np.exp(-np.square(x)/(2.0*ss))
    df["src_proba"] = df["src_proba"] / df["src_proba"].sum()
    
def get_trg_proba(df, eps, providers):
    df["trg_proba"] = eps + (1.0 - eps) * df.apply(lambda x: x["degree"] if x["pub_key"] in providers else 0.0, axis=1)
    df["trg_proba"] = df["trg_proba"] / df["trg_proba"].sum()
    
def sample_transactions(node_variables, amount_in_satoshi, K):
    nodes = list(node_variables["pub_key"])
    src_selected = np.random.choice(nodes, size=K, replace=True, p=list(node_variables["src_proba"]))
    trg_selected = np.random.choice(nodes, size=K, replace=True, p=list(node_variables["trg_proba"]))
    transactions = pd.DataFrame(list(zip(src_selected, trg_selected)), columns=["source","target"])
    transactions["amount_SAT"] = amount_in_satoshi
    transactions["transaction_id"] = transactions.index
    transactions = transactions[transactions["source"] != transactions["target"]]
    print("Number of loop transactions (removed):", K-len(transactions))
    return transactions[["transaction_id","source","target","amount_SAT"]]

