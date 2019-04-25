import json
import pandas as pd
import networkx as nx
import scipy.stats as st
import matplotlib.pyplot as plt

def load_temp_data(json_files, node_keys=["pub_key","last_update"], edge_keys=["node1_pub","node2_pub","last_update","capacity"]):
    """Load LN graph json files from several snapshots"""
    node_info, edge_info = [], []
    last_node_update, last_edge_update = 0, 0
    for json_f in json_files:
        with open(json_f) as f:
            tmp_json = json.load(f)
        new_nodes = pd.DataFrame(tmp_json["nodes"])[node_keys]
        new_edges = pd.DataFrame(tmp_json["edges"])[edge_keys]
        #print(len(new_edges))
        FILTER_TIME = 2524611600 #Human time (GMT): Saturday, January 1, 2050 1:00:00 AM
        new_edges = new_edges[new_edges["last_update"] < FILTER_TIME]
        #print(len(new_edges))
        new_nodes = new_nodes[new_nodes["last_update"] > last_node_update]
        new_edges = new_edges[new_edges["last_update"] > last_edge_update]
        print(json_f, len(new_nodes), len(new_edges))
        node_info.append(new_nodes)
        edge_info.append(new_edges)
        last_node_update = new_nodes["last_update"].max()
        last_edge_update = new_edges["last_update"].max()
    return pd.concat(node_info), pd.concat(edge_info)

def get_snapshots(edges_df, min_time, max_time, with_time=False, time_window=86400):
    """Split the LN network edges into snapshots based on the provided time window"""
    snapshots, current_snapshot = [], []
    snapshot_start, idx = min_time, 0
    for src, trg, time in list(zip(edges_df["node1_pub"],edges_df["node2_pub"],edges_df["last_update"])):
        if time > snapshot_start + time_window:
            snapshots.append(current_snapshot)
            print(idx, len(current_snapshot))
            current_snapshot = []
            snapshot_start += time_window
            idx += 1
            if snapshot_start > max_time:
                break
        if with_time:
            current_snapshot.append((src,trg,time))
        else:
            current_snapshot.append((src,trg))
    return snapshots

def get_snapshot_properties(snapshots, window=1, is_directed=True):
    """Calculate network properties for each snapshot"""
    stats = []
    for idx in range(len(snapshots)):
        from_idx = max(0,idx-window)
        window_edges = []
        for j in range(from_idx, idx+1):
            window_edges += snapshots[j]
        if is_directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_edges_from(window_edges)
        print(G.number_of_edges(), G.number_of_nodes())
        stats.append(calculate_centralities(G))
    return stats
            
def calculate_centralities(G):
    """Calculate centrality measures"""
    res = {
        "deg": nx.degree_centrality(G),
        "in_deg": nx.in_degree_centrality(G),
        "out_deg": nx.out_degree_centrality(G),
        "pr": nx.pagerank(G),
        "betw": nx.betweenness_centrality(G, k=None),
        "harm": nx.harmonic_centrality(G)
    }
    print("Centralities COMPUTED")
    return pd.DataFrame(res).reset_index()

def calc_corr(df, cent, corr_type="pearson"):
    """Calculate correlation for adjacent snapshot descriptors"""
    if corr_type == "spearman":
        return st.spearmanr(df[cent + "_0"], df[cent + "_1"])[0]
    elif corr_type == "kendall":
        return st.kendalltau(df[cent + "_0"], df[cent + "_1"])[0]
    elif corr_type == "w_kendall":
        return st.weightedtau(df[cent + "_0"], df[cent + "_1"])[0]
    else:
        return st.pearsonr(df[cent + "_0"], df[cent + "_1"])[0]
    
def get_corr_sequence(stats, corr_type):
    res = {"deg":[],"in_deg":[],"out_deg":[],"pr":[],"betw":[],"harm":[]}
    for idx in range(1,len(stats)):
        df1 = stats[idx-1]
        df2 = stats[idx]
        merged_df = df1.merge(df2, on="index", suffixes=("_0","_1"))
        merged_df = merged_df.fillna(0.0)
        for cent in  ["deg","in_deg","out_deg","pr","betw","harm"]:
            res[cent].append(calc_corr(merged_df, cent, corr_type))
    return res

def show_corr_results(results, cent):
    plt.Figure(figsize=(10,10))
    plt.title(cent)
    for corr in ["pearson","spearman","kendall","w_kendall"]:
        vals = results[corr][cent]
        plt.plot(range(len(vals)), vals, label=corr)
    plt.legend()
    plt.show()