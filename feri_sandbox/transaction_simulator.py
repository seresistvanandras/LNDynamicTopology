import pandas as pd
import numpy as np
import networkx as nx

def get_src_proba(df, alpha):
    df["src_proba"] = alpha / (alpha + df["degree"])
    df["src_proba"] = df["src_proba"] / df["src_proba"].sum()
    
def get_trg_proba(df, eps, providers):
    df["trg_proba"] = eps + (1.0 - eps) * df.apply(lambda x: x["degree"] if x["pub_key"] in providers else 0.0, axis=1)
    df["trg_proba"] = df["trg_proba"] / df["trg_proba"].sum()
    
def init_node_params(G, providers, eps, alpha=None):
    node_variables = pd.DataFrame(list(G.degree()), columns=["pub_key","degree"])
    if alpha == None:
        alpha = node_variables["degree"].mean()
    get_src_proba(node_variables, alpha)
    get_trg_proba(node_variables, eps, providers)
    return node_variables
    
def sample_transactions(node_variables, K=1000):
    nodes = list(node_variables["pub_key"])
    src_selected = np.random.choice(nodes, size=K, replace=True, p=list(node_variables["src_proba"]))
    trg_selected = np.random.choice(nodes, size=K, replace=True, p=list(node_variables["trg_proba"]))
    transactions = pd.DataFrame(list(zip(src_selected, trg_selected)), columns=["source","target"])
    transactions["amount"] = 1
    transactions["transaction_id"] = transactions.index
    transactions = transactions[transactions["source"] != transactions["target"]]
    print("Number of loop transactions (removed):", K-len(transactions))
    return transactions[["transaction_id","source","target","amount"]]

def get_shortest_paths(G, transactions, edges, hash_transactions=True, cost_prefix="", weight=None):
    keys = list(zip(edges["src"], edges["trg"]))
    vals = list(zip(edges["fee_base_msat"], edges["fee_rate_milli_msat"]))
    cost_dict = dict(zip(keys,vals))
    shortest_paths = []
    hashed_transactions = {}
    for idx, row in transactions.iterrows():
        try:
            p = nx.shortest_path(G, source=row["source"], target=row["target"], weight=weight)
            cost, routers = process_path(p, row["amount"]*10**8, cost_dict)
            if hash_transactions:
                for router in routers:
                    if not router in hashed_transactions:
                        hashed_transactions[router] = []
                    hashed_transactions[router].append(row)
        except:
            p = []
            cost = None
        finally:
            shortest_paths.append((row["transaction_id"], cost, len(p)-1, p))
    if hash_transactions:
        for node in hashed_transactions:
            hashed_transactions[node] = pd.DataFrame(hashed_transactions[node], columns=transactions.columns)
    return pd.DataFrame(shortest_paths, columns=["transaction_id", cost_prefix+"cost", "length", "path"]), hashed_transactions

def process_path(path, amount_in_satoshi, cost_dict):
    base_sum, rate_sum = 0.0, 0.0
    routers = []
    for i in range(len(path)-2):
        n1, n2 = path[i], path[i+1]
        new_base, new_rate = cost_dict[(n1,n2)]
        base_sum += new_base
        rate_sum += new_rate
        routers.append(n2)
    return base_sum + amount_in_satoshi * rate_sum / 10**6, routers

# TODO: parallelize!!!
def get_shortest_paths_with_node_removals(G, hashed_transactions, edges, cost_prefix="", weight=None):
    bin_sizes = []
    alternative_paths = []
    for node, bucket_transactions in hashed_transactions.items():
        bin_sizes.append(len(bucket_transactions))
        H = G.copy()
        H.remove_node(node)
        new_paths, _ = get_shortest_paths(H, bucket_transactions, edges, hash_transactions=False, cost_prefix=cost_prefix, weight=weight)
        new_paths["removed_node"] = node
        alternative_paths.append(new_paths)
    return pd.concat(alternative_paths), bin_sizes

def calculate_node_influence(shortest_paths, alternative_paths):
    s_paths = shortest_paths.copy().drop("path", axis=1)
    a_paths = alternative_paths.copy().drop("path", axis=1)
    s_paths["original_cost"] = 1.0 / s_paths["original_cost"]
    a_paths["cost"] = 1.0 / a_paths["cost"]
    s_paths["length"] = 1.0 / s_paths["length"]
    a_paths["length"] = 1.0 / a_paths["length"]
    routing_diff = a_paths.merge(s_paths, on="transaction_id", how="left", suffixes=("","_original"))
    routing_diff = routing_diff.fillna(0.0)
    harmonic_sums = routing_diff.drop("transaction_id", axis=1).groupby(by="removed_node").aggregate({"cost":"sum","original_cost":"sum"})
    harmonic_sums["cost_diff"] = harmonic_sums["original_cost"] - harmonic_sums["cost"]
    return harmonic_sums.sort_values("cost_diff", ascending=False), routing_diff

class TransactionSimulator():
    def __init__(self, edges, providers, k, eps=0.05, alpha=2.0):
        self.G = nx.from_pandas_dataframe(edges, source="src", target="trg", edge_attr=["capacity","fee_base_msat","fee_rate_milli_msat"])
        self.edges = edges
        self.providers = list(set(providers).intersection(set(self.G.nodes())))
        self.node_variables = init_node_params(self.G, self.providers, eps, alpha)
        print("Number of nodes:", self.G.number_of_nodes())
        print("Number of edges:", self.G.number_of_edges())
        self.transactions = sample_transactions(self.node_variables, k)
        print("%i transaction were generated." % k)
    
    def simulate(self):
        print("Transactions simulated on original graph STARTED..")
        shortest_paths, hashed_transactions = get_shortest_paths(self.G, self.transactions, self.edges, cost_prefix="original_")
        print("Transactions simulated on original graph DONE")
        print("Length distribution of optimal paths:")
        print(shortest_paths["length"].value_counts())
        print("Transactions simulated with node removals STARTED..")
        alternative_paths, bin_sizes = get_shortest_paths_with_node_removals(self.G, hashed_transactions, self.edges)
        print("Transactions simulated with node removals DONE")
        print("Length distribution of optimal paths:")
        print(alternative_paths["length"].value_counts())
        return shortest_paths, alternative_paths