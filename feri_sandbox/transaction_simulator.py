import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import os

### transaction simulator ###

def get_src_proba(df, alpha):
    df["src_proba"] = alpha / (alpha + df["degree"])
    df["src_proba"] = df["src_proba"] / df["src_proba"].sum()
    
def get_trg_proba(df, eps, providers):
    df["trg_proba"] = eps + (1.0 - eps) * df.apply(lambda x: x["degree"] if x["pub_key"] in providers else 0.0, axis=1)
    df["trg_proba"] = df["trg_proba"] / df["trg_proba"].sum()
    
def init_node_params(edges, providers, eps, alpha=None):
    G = nx.from_pandas_dataframe(edges, source="src", target="trg", create_using=nx.DiGraph())
    node_variables = pd.DataFrame(list(G.degree()), columns=["pub_key","degree"])
    if alpha == None:
        alpha = node_variables["degree"].mean()
    get_src_proba(node_variables, alpha)
    get_trg_proba(node_variables, eps, providers)
    return node_variables
    
def sample_transactions(node_variables, amount_in_satochi, K):
    nodes = list(node_variables["pub_key"])
    src_selected = np.random.choice(nodes, size=K, replace=True, p=list(node_variables["src_proba"]))
    trg_selected = np.random.choice(nodes, size=K, replace=True, p=list(node_variables["trg_proba"]))
    transactions = pd.DataFrame(list(zip(src_selected, trg_selected)), columns=["source","target"])
    transactions["amount_SAT"] = amount_in_satochi
    transactions["transaction_id"] = transactions.index
    transactions = transactions[transactions["source"] != transactions["target"]]
    print("Number of loop transactions (removed):", K-len(transactions))
    return transactions[["transaction_id","source","target","amount_SAT"]]

def get_shortest_paths(G, transactions, cost_dict, hash_transactions=True, cost_prefix="", weight=None):
    shortest_paths = []
    router_fee_tuples = []
    hashed_transactions = {}
    for idx, row in transactions.iterrows():
        try:
            p = nx.shortest_path(G, source=row["source"], target=row["target"] + "_trg", weight=weight)
            #c = nx.shortest_path_length(G, source=row["source"], target=row["target"] + "_trg", weight=weight)
            cost, router_fees = process_path(p, row["amount_SAT"], cost_dict)
            routers = list(router_fees.keys())
            router_fee_tuples += list(zip([row["transaction_id"]]*len(router_fees),router_fees.keys(),router_fees.values()))
            #if c != cost:
            #    raise RuntimeError("%i: %s->%s: %f - %f | %s" % (row["transaction_id"], row["source"], row["target"], c, cost, p))
            if hash_transactions:
                for router in routers:
                    if not router in hashed_transactions:
                        hashed_transactions[router] = []
                    hashed_transactions[router].append(row)
        except RuntimeError as re:
            raise re
        except:
            p = []
            cost = None
        finally:
            shortest_paths.append((row["transaction_id"], cost, len(p)-1, p))
    if hash_transactions:
        for node in hashed_transactions:
            hashed_transactions[node] = pd.DataFrame(hashed_transactions[node], columns=transactions.columns)
    all_router_fees = pd.DataFrame(router_fee_tuples, columns=["transaction_id","node","fee"])
    return pd.DataFrame(shortest_paths, columns=["transaction_id", cost_prefix+"cost", "length", "path"]), hashed_transactions,  all_router_fees

def process_path(path, amount_in_satoshi, cost_dict):
    routers = {}
    for i in range(len(path)-2):
        n1, n2 = path[i], path[i+1]
        base_fee, fee_rate = cost_dict[(n1,n2)]
        routers[n2] = base_fee / 1000.0 + amount_in_satoshi * fee_rate / 10.0**6
    return np.sum(list(routers.values())), routers

def shortest_paths_with_exclusion(G, cost_dict, cost_prefix, weight, hash_bucket_item):
    node, bucket_transactions = hash_bucket_item
    H = G.copy()
    H.remove_node(node)
    H.remove_node(node + "_trg") # delete node copy as well
    new_paths, _, _ = get_shortest_paths(H, bucket_transactions, cost_dict, hash_transactions=False, cost_prefix=cost_prefix, weight=weight)
    new_paths["node"] = node
    return new_paths

import functools
import concurrent.futures

def get_shortest_paths_with_node_removals(G, hashed_transactions, cost_dict, cost_prefix="", weight=None, threads=4):
    print(threads)
    if threads > 1:
        f_partial = functools.partial(shortest_paths_with_exclusion, G, cost_dict, cost_prefix, weight)
        executor = concurrent.futures.ProcessPoolExecutor(threads)
        alternative_paths = list(executor.map(f_partial, hashed_transactions.items()))
        executor.shutdown()
    else:
        alternative_paths = []
        for hash_bucket_item in tqdm(hashed_transactions.items(), mininterval=10):
            alternative_paths.append(shortest_paths_with_exclusion(G, cost_dict, cost_prefix, weight, hash_bucket_item))
    return pd.concat(alternative_paths)

def generate_graph_for_path_search(edges):
    """The last edge in each path has zero cost! Only routing has transaction fees."""
    tmp_edges = edges.copy()
    tmp_edges["trg"] = tmp_edges["trg"].apply(lambda x: str(x) + "_trg")
    tmp_edges["total_fee"] = 0.0
    tmp_edges["fee_base_msat"] = 0.0
    tmp_edges["fee_rate_milli_msat"] = 0.0
    all_edges = pd.concat([edges, tmp_edges])
    return nx.from_pandas_dataframe(all_edges, source="src", target="trg", edge_attr=["total_fee"], create_using=nx.DiGraph())

class TransactionSimulator():
    def __init__(self, edges, providers, amount_sat, k, eps=0.05, alpha=2.0):
        self.edges = edges[edges["capacity"] > amount_sat]
        print("Number of deleted directed channels:", len(edges) - len(self.edges))
        print("Number of remaining directed channels:", len(self.edges))
        self.edges["total_fee"] = self.edges["fee_base_msat"] / 1000.0 + amount_sat * self.edges["fee_rate_milli_msat"] / 10.0**6
        self.G = generate_graph_for_path_search(self.edges)
        self.providers = list(set(providers).intersection(set(self.G.nodes())))
        self.node_variables = init_node_params(self.edges, self.providers, eps, alpha)
        self.transactions = sample_transactions(self.node_variables, amount_sat, k)
        print("%i transaction were generated." % k)
    
    def simulate(self, weight=None, max_threads=4):
        print("Using weight='%s'" % weight)
        keys = list(zip(self.edges["src"], self.edges["trg"]))
        vals = list(zip(self.edges["fee_base_msat"], self.edges["fee_rate_milli_msat"]))
        cost_dict = dict(zip(keys,vals))
        print("Transactions simulated on original graph STARTED..")
        shortest_paths, hashed_transactions, all_router_fees = get_shortest_paths(self.G, self.transactions, cost_dict, cost_prefix="original_", weight=weight)
        print("Transactions simulated on original graph DONE")
        print("Length distribution of optimal paths:")
        print(shortest_paths["length"].value_counts())
        print("Transactions simulated with node removals STARTED..")
        alternative_paths = get_shortest_paths_with_node_removals(self.G, hashed_transactions, cost_dict, weight=weight, threads=max_threads)
        print("Transactions simulated with node removals DONE")
        print("Length distribution of optimal paths:")
        print(alternative_paths["length"].value_counts())
        return shortest_paths, alternative_paths, all_router_fees
    
### process results ###

def calculate_node_influence(shortest_paths, alternative_paths):
    s_paths = shortest_paths.copy().drop("path", axis=1)
    a_paths = alternative_paths.copy().drop("path", axis=1)
    s_paths["original_cost"] = 1.0 / (1.0 + s_paths["original_cost"])
    a_paths["cost"] = 1.0 / (1.0 + a_paths["cost"])
    routing_diff = a_paths.merge(s_paths, on="transaction_id", how="left", suffixes=("","_original"))
    routing_diff = routing_diff.fillna(0.0)
    harmonic_sums = routing_diff.drop("transaction_id", axis=1).groupby(by="node").aggregate({"cost":"sum","original_cost":"sum"})
    harmonic_sums["cost_diff"] = harmonic_sums["original_cost"] - harmonic_sums["cost"]
    return harmonic_sums.sort_values("cost_diff", ascending=False), routing_diff

def get_experiment_files(experiment_id, snapshots, simulation_dir):
    files = {}
    for snap_id in snapshots:
        files[snap_id] = []
        for f in os.listdir("%s/%i" % (simulation_dir, snap_id)):
            if experiment_id in f:
                files[snap_id].append("%s/%i/%s" % (simulation_dir, snap_id, f))
    return files

def aggregate_samples(experiment_files, snapshot_id):
    samples = []
    for i, f in enumerate(experiment_files[snapshot_id]):
        df = pd.read_csv(f)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df[~df["cost_diff"].isnull()]
        df["sample_id"] = i
        samples.append(df)
    df = pd.concat(samples, sort=True)
    mean_costs = df.groupby("node").mean().drop("sample_id", axis=1)
    return merge_with_other_metrics(mean_costs.sort_values("cost_diff", ascending=False), snapshot_id), df

def merge_with_other_metrics(mean_costs, snapshot_id, weight=None):
    cent = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/centrality_scores/scores_%s_%i.csv" % (weight, snapshot_id))
    most_pop = pd.read_csv("/mnt/idms/fberes/data/bitcoin_ln_research/most_pop_nodes.csv")
    all_info = mean_costs.reset_index().merge(cent[["index","betw","deg","pr"]], left_on="node", right_on="index", how="left").drop("index", axis=1)
    all_info = all_info.merge(most_pop[["index",str(snapshot_id)]], left_on="node", right_on="index", how="left").drop("index", axis=1)
    all_info = all_info.rename({str(snapshot_id):"pop"}, axis=1)
    all_info = all_info.fillna(0)
    return all_info