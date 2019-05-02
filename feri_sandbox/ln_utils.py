import json
import pandas as pd
import networkx as nx
import scipy.stats as st
import matplotlib.pyplot as plt

def load_temp_data(json_files, node_keys=["pub_key","last_update"], edge_keys=["node1_pub","node2_pub","last_update","capacity"]):
    """Load LN graph json files from several snapshots"""
    node_info, edge_info = [], []
    last_node_update, last_edge_update = 0, 0
    first = True
    for json_f in json_files:
        with open(json_f) as f:
            tmp_json = json.load(f)
        new_nodes = pd.DataFrame(tmp_json["nodes"])[node_keys]
        new_edges = pd.DataFrame(tmp_json["edges"])[edge_keys]
        """
        if first:
            nodes = new_nodes
            edges = new_edges
            first = False
        else:
            nodes = pd.concat([nodes,new_nodes]).drop_duplicates()
            edges = pd.concat([edges,new_edges]).drop_duplicates()
            print(json_f, len(nodes), len(edges))
        """
        #print(len(new_edges))
        FILTER_TIME = 2524611600 #Human time (GMT): Saturday, January 1, 2050 1:00:00 AM
        new_edges = new_edges[new_edges["last_update"] < FILTER_TIME]
        #print(len(new_edges))
        new_nodes = new_nodes[new_nodes["last_update"] > last_node_update]
        new_edges = new_edges[new_edges["last_update"] > last_edge_update]
        print(json_f, len(new_nodes), len(new_edges))
        if len(new_nodes) > 0:
            node_info.append(new_nodes)
            last_node_update = new_nodes["last_update"].max()
        else:
            print("NO NEW NODES!!!")
        if len(new_edges) > 0:
            edge_info.append(new_edges)
            last_edge_update = new_edges["last_update"].max()
        else:
            print("NO NEW EDGES!!!")
        #print(last_node_update, last_edge_update)
    #return nodes, edges
    return pd.concat(node_info), pd.concat(edge_info)


def get_snapshots(edges_df, min_time, max_time, with_data=False, time_window=86400):
    """Split the LN network edges into snapshots based on the provided time window"""
    snapshots, current_snapshot = [], []
    snapshot_start, idx = min_time, 0
    for src, trg, time, cap in list(zip(edges_df["node1_pub"],edges_df["node2_pub"],edges_df["last_update"], edges_df["capacity"])):
        if time > snapshot_start + time_window:
            snapshots.append(current_snapshot)
            print(idx, len(current_snapshot))
            current_snapshot = []
            snapshot_start += time_window
            idx += 1
            if snapshot_start > max_time:
                break
        if with_data:
            current_snapshot.append((src,trg,{'capacity':int(cap),'last_update':int(time)}))
        else:
            current_snapshot.append((src,trg))
    return snapshots

def get_snapshot_properties(snapshots, window=1, is_directed=False, weight=None):
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
        stats.append(calculate_centralities(G,weight))
    return stats
            
def calculate_centralities(G,weight=None):
    """Calculate centrality measures"""
    res = {
        "deg": nx.degree_centrality(G),
        "pr": nx.pagerank(G,weight=weight),
        "betw": nx.betweenness_centrality(G, k=None, weight=weight),
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
    res = {"deg":[],"pr":[],"betw":[],"harm":[]}
    for idx in range(1,len(stats)):
        df1 = stats[idx-1]
        df2 = stats[idx]
        merged_df = df1.merge(df2, on="index", suffixes=("_0","_1"))
        merged_df = merged_df.fillna(0.0)
        for cent in  ["deg","pr","betw","harm"]:
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