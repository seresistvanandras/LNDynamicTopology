import json
import pandas as pd
import numpy as np
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
    edges = pd.concat(edge_info)
    edges["capacity"] = edges["capacity"].astype("int64")
    edges["last_update"] = edges["last_update"].astype("int64")
    edges_no_loops = edges[edges["node1_pub"] != edges["node2_pub"]]
    return pd.concat(node_info), edges_no_loops

### network centrality analysis ###

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
            current_snapshot.append((src,trg,{'capacity':cap,'last_update':time}))
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
        "deg": dict(G.degree(weight=None)),
        "wdeg": dict(G.degree(weight=weight)),
        "pr": nx.pagerank(G, weight=weight),
        "betw": nx.betweenness_centrality(G),#, weight=weight, k=None),
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
    
def get_corr_sequence(stats, corr_type, adjacent=True):
    res = {"deg":[],"wdeg":[],"pr":[],"betw":[],"harm":[]}
    for idx in range(1,len(stats)):
        if adjacent:
            df1 = stats[idx-1]
        else:
            df1 = stats[0]
        df2 = stats[idx]
        merged_df = df1.merge(df2, on="index", suffixes=("_0","_1"))
        merged_df = merged_df.fillna(0.0)
        for cent in  ["deg","wdeg","pr","betw","harm"]:
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
    
### node attachments ###

def calc_centralities(G):
    print("Calculate centralities STARTED")
    degree = dict(nx.degree(G))
    w_degree = dict(G.degree(weight="capacity"))
    betw = nx.betweenness_centrality(G, weight=None)
    betw_cap = nx.betweenness_centrality(G, weight="rec_cap")
    pr = nx.pagerank(G, weight="capacity")
    betw_rank = dict(zip(betw.keys(),st.rankdata(-np.array(list(betw.values())))))
    betw_cap_rank = dict(zip(betw_cap.keys(),st.rankdata(-np.array(list(betw_cap.values())))))
    degree_rank = dict(zip(degree.keys(),st.rankdata(-np.array(list(degree.values())))))
    w_degree_rank = dict(zip(w_degree.keys(),st.rankdata(-np.array(list(w_degree.values())))))
    pr_rank = dict(zip(pr.keys(),st.rankdata(-np.array(list(pr.values())))))
    print("Calculate centralities FINISHED")
    scores = {"deg":degree, "wdeg":w_degree, "betw": betw, "betw_cap":betw_cap, "pr":pr}
    ranks = {"deg":degree_rank, "wdeg":w_degree_rank, "betw": betw_rank, "betw_cap":betw_cap_rank, "pr":pr_rank}
    return scores, ranks

def analyse_last_snapshot(last_snap_file, edge_keys):
    init_nodes, init_edges = load_temp_data([last_snap_file], edge_keys=edge_keys)
    last_time = init_edges["last_update"].max()
    print(len(init_nodes), len(init_edges), last_time)
    N = set(init_nodes["pub_key"])
    E = init_edges[["node1_pub","node2_pub","channel_id","capacity"]].drop_duplicates()
    E = E.groupby(["node1_pub","node2_pub"])["capacity"].sum().reset_index()
    E["rec_cap"] = 1.0 / E["capacity"]
    G = nx.from_pandas_dataframe(E, source="node1_pub", target="node2_pub", edge_attr=["capacity","rec_cap"], create_using=nx.Graph())
    print(G.number_of_nodes(), G.number_of_edges())
    scores, ranks = calc_centralities(G)
    return N, E, scores, ranks, last_time

def get_new_node_attachements(edges, known_nodes, node_ranks):
    new_node_records = []
    homophily_edges = []
    for idx, row in edges.iterrows():
        n1, n2, t = row["node1_pub"], row["node2_pub"], row["last_update"]
        b1 = n1 in known_nodes
        b2 = n2 in known_nodes
        if b1 and not b2:
            new, old = n2, n1
        elif b2 and not b1:
            new, old = n1, n2
        else:
            homophily_edges.append((n1,n2))
            continue
        new_node_records.append((t, new, old, node_ranks["betw"][old], node_ranks["betw_cap"][old], node_ranks["deg"][old], node_ranks["pr"][old], node_ranks["wdeg"][old]))
    attachments_df = pd.DataFrame(new_node_records, columns=["time", "new_node", "old_node", "betw_rank", "betw_cap_rank", "degree_rank", "pr_rank", "w_degree_rank"])
    return attachments_df, homophily_edges

def observe_node_attachements_over_time(files, first_index=1, window=7, edge_keys=["node1_pub","node2_pub","last_update","capacity","channel_id"]):
    num_files = len(files)
    idx = first_index
    attachments, node_list, scores_list, ranks_list = [], [], [], []
    while idx < num_files:
        # process last snapshot
        N, _, scores, ranks, _ = analyse_last_snapshot(files[idx-1], edge_keys)
        # process the last time window
        nodes, edges = load_temp_data(files[idx:idx+window], edge_keys=edge_keys)
        # discover new node attachements
        attachments_df, homophily = get_new_node_attachements(edges, N, ranks)
        attachments.append(attachments_df)
        node_list.append(N)
        scores_list.append(scores)
        ranks_list.append(ranks)
        idx += window
    print()
    return attachments, node_list, scores_list, ranks_list

def get_attachement_popularity(attachments):
    pops = [att["old_node"].value_counts() for att in attachments]
    pop_df = pd.DataFrame(pops).T
    pop_df.columns = range(pop_df.shape[1])
    pop_df = pop_df.fillna(0.0)
    return pop_df

def corr_mx(df, method="spearman"):
    s = df.shape[1]
    arr = df.values
    corr = np.ones((s,s))
    for i in range(s):
        for j in range(i+1,s):
            if method == "wkendall":
                corr[i,j] = st.weightedtau(arr[:,i],arr[:,j])[0]
            elif method == "kendall":
                corr[i,j] = st.kendalltau(arr[:,i],arr[:,j])[0]
            else:
                corr[i,j] = st.spearmanr(arr[:,i],arr[:,j])[0]
            corr[j,i] = corr[i,j]
    return pd.DataFrame(corr, columns=df.columns)

def pop_corr_with_centralities(pop_df, cent_scores, method="spearman", cent_keys=["betw","betw_cap","wdeg"]):
    node_pubs = list(pop_df.index)
    time_series = dict((cent,[]) for cent in cent_keys)
    for i in range(len(cent_scores)):
        popvals = list(pop_df[i])
        for cent in cent_keys:
            centvals = [cent_scores[i][cent].get(n,0.0) for n in node_pubs]
            if method == "wkendall":
                time_series[cent].append(st.weightedtau(popvals, centvals)[0])
            elif method == "kendall":
                time_series[cent].append(st.kendalltau(popvals, centvals)[0])
            else:
                time_series[cent].append(st.spearmanr(popvals, centvals)[0])
    return time_series