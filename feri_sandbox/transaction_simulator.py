import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy

### transaction simulator ###

#### sampling transactions ###

def get_src_proba(df, alpha):
    df["src_proba"] = alpha / (alpha + df["degree"])
    df["src_proba"] = df["src_proba"] / df["src_proba"].sum()
    
def get_src_rayleigh_proba(df):
    s = df["total_capacity"].median()
    print(s)
    x = np.array(df["total_capacity"], dtype="float64")
    df["src_proba"] = (x/s**2) * np.exp(-np.square(x)/(2.0*s**2))
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
    
#### shortest path search ####

def get_shortest_paths(init_capacities, G_origi, transactions, hash_transactions=True, cost_prefix="", weight=None):
    G = G_origi.copy()# copy due to forthcoming graph capacity changes!!!
    capacity_map = copy.deepcopy(init_capacities)#init_capacities.copy()
    shortest_paths = []
    router_fee_tuples = []
    hashed_transactions = {}
    for idx, row in transactions.iterrows():
        p, cost = [], None
        try:
            p = nx.shortest_path(G, source=row["source"], target=row["target"] + "_trg", weight=weight)
            if row["target"] in p:
                raise RuntimeError("Loop detected: %s" % row["target"])
            cost, router_fees = process_path(p, row["amount_SAT"], capacity_map, G, weight)
            routers = list(router_fees.keys())
            router_fee_tuples += list(zip([row["transaction_id"]]*len(router_fees),router_fees.keys(),router_fees.values()))
            if hash_transactions:
                for router in routers:
                    if not router in hashed_transactions:
                        hashed_transactions[router] = []
                    hashed_transactions[router].append(row)
        except nx.NetworkXNoPath:
            continue
        except:
            print(idx)
            print(p)
            raise
        finally:
            shortest_paths.append((row["transaction_id"], cost, len(p)-1, p))
    if hash_transactions:
        for node in hashed_transactions:
            hashed_transactions[node] = pd.DataFrame(hashed_transactions[node], columns=transactions.columns)
    all_router_fees = pd.DataFrame(router_fee_tuples, columns=["transaction_id","node","fee"])
    return pd.DataFrame(shortest_paths, columns=["transaction_id", cost_prefix+"cost", "length", "path"]), hashed_transactions,  all_router_fees

def process_path(path, amount_in_satoshi, capacity_map, G, weight):
    routers = {}
    N = len(path)
    for i in range(N-2):
        n1, n2 = path[i], path[i+1]
        routers[n2] = G[n1][n2][weight]
        _ = process_forward_edge(capacity_map, G, amount_in_satoshi, n1, n2)   
        _ = process_backward_edge(capacity_map, G, amount_in_satoshi, n2, n1)
    n1, n2 = path[N-2], path[N-1].replace("_trg","")
    _ = process_forward_edge(capacity_map, G, amount_in_satoshi, n1, n2)
    _ = process_backward_edge(capacity_map, G, amount_in_satoshi, n2, n1)
    return np.sum(list(routers.values())), routers

def process_forward_edge(capacity_map, G, amount_in_satoshi, src, trg):
    cap, fee, is_trg_provider, total_cap = capacity_map[(src,trg)]
    if cap < amount_in_satoshi:
        raise RuntimeError("forward %i: %s-%s" % (cap,src,trg))
    if cap < 2*amount_in_satoshi: # cannot route more transactions
        G.remove_edge(src, trg)
        if is_trg_provider:
            G.remove_edge(src, trg+'_trg')
    capacity_map[(src,trg)] = [cap-amount_in_satoshi, fee, is_trg_provider, total_cap]
    
def process_backward_edge(capacity_map, G, amount_in_satoshi, src, trg):
    if (src,trg) in capacity_map:
        cap, fee, is_trg_provider, total_cap = capacity_map[(src,trg)]
        if cap < amount_in_satoshi: # it can route transactions again
            G.add_weighted_edges_from([(src,trg,fee)], weight="total_fee")
            if is_trg_provider:
                G.add_weighted_edges_from([(src,trg+'_trg',0.0)], weight="total_fee")
        capacity_map[(src,trg)] = [cap+amount_in_satoshi, fee, is_trg_provider, total_cap]

def shortest_paths_with_exclusion(capacity_map, G, cost_prefix, weight, hash_bucket_item):
    node, bucket_transactions = hash_bucket_item
    H = G.copy()
    H.remove_node(node)
    if node + "_trg" in G.nodes():
        H.remove_node(node + "_trg") # delete node copy as well
    new_paths, _, _ = get_shortest_paths(capacity_map, H, bucket_transactions,  hash_transactions=False, cost_prefix=cost_prefix, weight=weight)
    new_paths["node"] = node
    return new_paths

import functools
import concurrent.futures

def get_shortest_paths_with_node_removals(capacity_map, G, hashed_transactions, cost_prefix="", weight=None, threads=4):
    print(threads)
    if threads > 1:
        f_partial = functools.partial(shortest_paths_with_exclusion, capacity_map, G, cost_prefix, weight)
        executor = concurrent.futures.ProcessPoolExecutor(threads)
        alternative_paths = list(executor.map(f_partial, hashed_transactions.items()))
        executor.shutdown()
    else:
        alternative_paths = []
        for hash_bucket_item in tqdm(hashed_transactions.items(), mininterval=10):
            alternative_paths.append(shortest_paths_with_exclusion(capacity_map, G, cost_prefix, weight, hash_bucket_item))
    return pd.concat(alternative_paths)

#### preprocess edges and graph ####

def init_node_params(edges, providers, eps, alpha=None):
    G = nx.from_pandas_edgelist(edges, source="src", target="trg", edge_attr=["capacity"], create_using=nx.DiGraph())
    providers = list(set(providers).intersection(set(G.nodes())))
    degrees = pd.DataFrame(list(G.degree()), columns=["pub_key","degree"])
    total_capacity = pd.DataFrame(list(nx.degree(G, weight="capacity")), columns=["pub_key","total_capacity"])
    node_variables = degrees.merge(total_capacity, on="pub_key")
    if alpha == None:
        get_src_rayleigh_proba(node_variables)
    else:
        get_src_proba(node_variables, alpha)
    get_trg_proba(node_variables, eps, providers)
    return node_variables, providers

def generate_graph_for_path_search(edges, transactions):
    """The last edge in each path has zero cost! Only routing has transaction fees."""
    targets = list(transactions["target"].unique())
    sources = list(transactions["source"].unique())
    participants = set(sources).union(set(targets))
    edges_tmp = edges.copy()
    # add pseudo targets
    ps_edges = edges[edges["trg"].isin(targets)].copy()
    ps_edges["trg"] = ps_edges["trg"].apply(lambda x: str(x) + "_trg")
    ps_edges["total_fee"] = 0.0
    ps_edges["fee_base_msat"] = 0.0
    ps_edges["fee_rate_milli_msat"] = 0.0
    # initialize transaction graph
    all_edges = pd.concat([edges_tmp, ps_edges])
    # networkx versiom >= 2: from_pandas_edgelist
    G = nx.from_pandas_edgelist(all_edges, source="src", target="trg", edge_attr=["total_fee"], create_using=nx.DiGraph())
    return G

def calculate_tx_fee(df, amount_sat, epsilon=10**-9):
    """epsilon is needed to have a positive weight function for path routing"""
    # first part: fee_base_msat -> fee_base_sat
    # second part: milli_msat == 10^-6 sat : fee_rate_milli_msat -> fee_rate_sat
    return epsilon + df["fee_base_msat"] / 1000.0 + amount_sat * df["fee_rate_milli_msat"] / 10.0**6

def prepare_edges_for_simulation(edges, amount_sat, drop_disabled):
    # remove edges with capacity below threshold
    tmp_edges = edges[edges["capacity"] > amount_sat]
    # remove disabled edges
    if drop_disabled:
        tmp_edges = tmp_edges[~tmp_edges["disabled"]]
    print("Number of deleted directed channels:", len(edges) - len(tmp_edges))
    print("Number of remaining directed channels:", len(tmp_edges))
    tmp_edges["total_fee"] = calculate_tx_fee(tmp_edges, amount_sat)
    print("Total transaction fee per edge were calculated")
    grouped = tmp_edges.groupby(["src","trg"])
    directed_aggr = grouped.agg({
        "capacity":"sum",
        "total_fee":"mean",
    }).reset_index()
    print("Number of edges after aggregation: %i" % len(directed_aggr))
    return directed_aggr

def init_capacities(edges, tx_targets, amount_in_sat):
    # init capacity dict
    keys = list(zip(edges["src"], edges["trg"]))
    is_trg_provider = edges["trg"].apply(lambda x: x in tx_targets)
    vals = [list(item) for item in zip([None]*len(edges), edges["total_fee"], is_trg_provider, edges["capacity"])]
    current_capacity_map = dict(zip(keys,vals))
    # extract channels
    channels = set()
    for s, t in keys:
        if (s,t) in channels or (t,s) in channels:
            continue
        else:
            channels.add((s,t))
    edges_with_capacity = populate_capacities(channels, current_capacity_map, amount_in_sat)
    print("Edges with capacity: %i->%i" % (len(edges),len(edges_with_capacity))) 
    return current_capacity_map, edges_with_capacity
    
def populate_capacities(channels, capacity_map, amount_in_sat):
    edge_records = []
    for src, trg in channels:
        c1 = capacity_map[(src,trg)][3]
        if (trg,src) in capacity_map:
            c2 = capacity_map[(trg,src)][3]
            cap = max(c1, c2)
            rnd = np.random.random()
            cap1, cap2 = cap * rnd, cap * (1.0-rnd) * cap
            capacity_map[(trg,src)][0] = cap2
            if cap2 >= amount_in_sat:
                edge_records.append((trg,src,cap2,capacity_map[(trg,src)][1]))
        else:
            cap1 = c1
        capacity_map[(src,trg)][0] = cap1
        if cap1 >= amount_in_sat:
                edge_records.append((src,trg,cap1,capacity_map[(src,trg)][1]))
    return pd.DataFrame(edge_records, columns=["src","trg","capacity","total_fee"])

class TransactionSimulator():
    def __init__(self, edges, providers, amount_sat, k, eps=0.05, alpha=2.0, drop_disabled=True):
        self.edges = prepare_edges_for_simulation(edges, amount_sat, drop_disabled)
        self.node_variables, self.providers = init_node_params(self.edges, providers, eps, alpha)
        self.transactions = sample_transactions(self.node_variables, amount_sat, k)
        self.current_capacity_map, self.edges_with_capacity = init_capacities(self.edges, set(self.transactions["target"]), amount_sat)
        self.G = generate_graph_for_path_search(self.edges_with_capacity, self.transactions)
        print("%i transactions were generated." % k)
    
    def simulate(self, weight=None, with_node_removals=True, max_threads=4):
        print("Using weight='%s'" % weight)
        print("Transactions simulated on original graph STARTED..")
        shortest_paths, hashed_transactions, all_router_fees = get_shortest_paths(self.current_capacity_map, self.G, self.transactions, cost_prefix="original_", weight=weight)
        print("Transactions simulated on original graph DONE")
        print("Length distribution of optimal paths:")
        print(shortest_paths["length"].value_counts())
        print("Transactions simulated with node removals STARTED..")
        if with_node_removals:
            alternative_paths = get_shortest_paths_with_node_removals(self.current_capacity_map, self.G, hashed_transactions, weight=weight, threads=max_threads)
            print("Transactions simulated with node removals DONE")
            print("Length distribution of optimal paths:")
            print(alternative_paths["length"].value_counts())
        else:
            alternative_paths = pd.DataFrame([])
        self.shortest_paths = shortest_paths
        self.alternative_paths = alternative_paths
        self.all_router_fees = all_router_fees
        return shortest_paths, alternative_paths, all_router_fees
    
    def export(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        total_income = get_total_income_for_routers(self.all_router_fees)
        total_income.to_csv("%s/router_incomes.csv" % output_dir, index=True)
        total_fee = get_total_fee_for_sources(self.transactions, self.shortest_paths)
        total_fee.to_csv("%s/source_fees.csv" % output_dir, index=True)
        if len(self.alternative_paths) > 0: 
            print(self.alternative_paths["cost"].isnull().value_counts())
        print("Export DONE")
        return total_income, total_fee
    
### process results ###

def get_total_income_for_routers(all_router_fees):
    return all_router_fees.groupby("node")["fee"].sum().sort_values(ascending=False)

def get_total_fee_for_sources(transactions, shortest_paths):
    trans_with_costs = transactions[["transaction_id","source"]].merge(shortest_paths[["transaction_id","original_cost"]], on="transaction_id")
    trans_with_costs = trans_with_costs[~trans_with_costs["original_cost"].isnull()]
    agg_funcs = dict(original_cost='mean', transaction_id='nunique')
    aggs = trans_with_costs.groupby(by="source")["original_cost"].agg(agg_funcs).rename({"original_cost":"mean_fee","transaction_id":"num_trans"}, axis=1)
    return aggs

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

### optimal fee pricing ###

def calculate_max_income(n, p_altered, shortest_paths, all_router_fees, visualize=False, min_ratio=0.0):
    trans = p_altered[p_altered["node"] == n]
    trans = trans.merge(shortest_paths, on="transaction_id", how="left")#'original_cost' column merged
    trans = trans.merge(all_router_fees, on=["transaction_id","node"], how="left")#'fee' column is merged
    trans["delta_cost"] = trans["cost"] - trans["original_cost"]
    ordered_deltas = trans[["transaction_id","fee","delta_cost"]].sort_values("delta_cost")
    ordered_deltas["delta_cost"] = ordered_deltas["delta_cost"].apply(lambda x: round(x, 2))
    thresholds = sorted(list(ordered_deltas["delta_cost"].unique()))
    original_income = ordered_deltas["fee"].sum()
    original_num_transactions = len(ordered_deltas)
    probas, incomes = [], []
    for th in thresholds:
        df = ordered_deltas[ordered_deltas["delta_cost"] >= th]
        prob = len(df) / original_num_transactions
        total = df["fee"].sum() + len(df) * th
        incomes.append(total)
        probas.append(prob)
        if prob < min_ratio:
            break
    max_idx = np.argmax(incomes)
    if visualize:
        fig, ax1 = plt.subplots()
        ax1.set_title(original_num_transactions)
        ax1.plot(thresholds[:len(incomes)], incomes, 'bx-')
        ax1.set_xscale("log")
        #ax1.plot(thresholds[:len(incomes)], np.array(incomes)*np.array(probas), 'rx-')
        ax2 = ax1.twinx()
        ax2.plot(thresholds[:len(incomes)], probas, 'gx-')
        ax2.set_xscale("log")
    return thresholds[max_idx], incomes[max_idx], probas[max_idx], original_income, original_num_transactions

def calc_optimal_base_fee(shortest_paths, alternative_paths, all_router_fees):
    alternative_paths_tmp = alternative_paths.copy()
    #alternative_paths_tmp["cost"] = alternative_paths["cost"].fillna(25000)
    p_altered = alternative_paths_tmp[~alternative_paths_tmp["cost"].isnull()]
    "Path ratio that have alternative routing after removals: %f" % (len(p_altered) / len(alternative_paths_tmp))
    num_routers = len(alternative_paths_tmp["node"].unique())
    num_routers_with_alternative_paths = len(p_altered["node"].unique())
    "Node ratio that have alternative routing after removals: %f" % (num_routers_with_alternative_paths / num_routers)
    routers = list(p_altered["node"].unique())
    opt_strategy = []
    for n in tqdm(routers, mininterval=5):
        opt_delta, opt_income, opt_ratio, origi_income, origi_num_trans = calculate_max_income(n, p_altered, shortest_paths, all_router_fees, visualize=False)
        opt_strategy.append((n, opt_delta, opt_ratio, opt_income, origi_income, origi_num_trans))
    opt_fees_df = pd.DataFrame(opt_strategy, columns=["node","opt_delta","opt_traffic","opt_income","origi_income", "origi_num_trans"])
    opt_fees_df["income_gain"] = ((opt_fees_df["opt_income"] - opt_fees_df["origi_income"]) / opt_fees_df["origi_income"]).replace(np.inf, 100.0)
    opt_fees_df = opt_fees_df.sort_values("origi_income", ascending=False)
    print("Mean fee pricing statistics:")
    print(opt_fees_df[["opt_delta","opt_traffic","origi_num_trans","income_gain"]].mean())
    return opt_fees_df, p_altered