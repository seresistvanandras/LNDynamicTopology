import json
import pandas as pd
import numpy as np
import networkx as nx
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

### preprocessing ###

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

def generate_directed_graph(edges, policy_keys=['disabled', 'fee_base_msat', 'fee_rate_milli_msat', 'min_htlc']):
    directed_edges = []
    for idx, row in edges.iterrows():
        e1 = [row[x] for x in ["node1_pub","node2_pub","last_update","channel_id","capacity"]]
        e2 = [row[x] for x in ["node2_pub","node1_pub","last_update","channel_id","capacity"]]
        if row["node2_policy"] == None:
            e1 += [None for x in policy_keys]
        else:
            e1 += [row["node2_policy"][x] for x in policy_keys]
        if row["node1_policy"] == None:
            e2 += [None for x in policy_keys]
        else:
            e2 += [row["node1_policy"][x] for x in policy_keys]
        directed_edges += [e1, e2]
    cols = ["src","trg","last_update","channel_id","capacity"] + policy_keys
    directed_edges_df = pd.DataFrame(directed_edges, columns=cols)
    return directed_edges_df

def load_centrality_scores(stat_dir, snapshot_ids, weight_cols, drop_cols=[]):
    stats = dict([(w,[]) for w in weight_cols])
    for w in weight_cols:
        for idx in snapshot_ids:
            snap_stat = pd.read_csv("%s/scores_%s_%i.csv" % (stat_dir, w, idx))
            if len(drop_cols) > 0:
                snap_stat = snap_stat.drop(drop_cols, axis=1)
            stats[w].append(snap_stat)
    return stats

### network centrality analysis ###

def get_snapshots(edges_df, min_time, max_time, time_window, weight_cols=None):
    """Split the LN network edges into snapshots based on the provided time window"""
    snapshot_graphs = []
    snapshot_edges = []
    L = (max_time-min_time) // time_window + 1
    for i in range(1,L+1):
        snap_edges = edges_df[edges_df["last_update"] < min_time+i*time_window]
        # drop channels without capacity (if any exists)
        snap_edges = snap_edges[snap_edges["capacity"]>0]
        snapshot_edges.append(snap_edges[snap_edges["last_update"] >= min_time+(i-1)*time_window])
        if weight_cols != None and "capacity" in weight_cols:
            # reciprocal capacity for betweeness centrality computation
            snap_edges["rec_capacity"] = 1.0 / snap_edges["capacity"]
            cols = weight_cols.copy()
            cols.append("rec_capacity")
        else:
            cols = weight_cols
        snap_edges = snap_edges.drop_duplicates(["src","trg"],  keep='last')
        print(i, len(snap_edges))
        snapshot_graphs.append(nx.from_pandas_dataframe(snap_edges, source="src", target="trg", edge_attr=cols, create_using=nx.DiGraph()))
    return snapshot_graphs, snapshot_edges

def get_snapshot_properties(snapshot_graphs, weight):
    """Calculate network properties for each snapshot"""
    stats = []
    for G in snapshot_graphs:
        print(G.number_of_edges(), G.number_of_nodes())
        stats.append(calculate_centralities(G, weight))
    return stats
            
def calculate_centralities(G, weight=None):
    """Calculate centrality measures"""
    res = {
        "deg": dict(G.degree(weight=weight)),
        "in_deg": dict(G.in_degree(weight=weight)),
        "out_deg": dict(G.out_degree(weight=weight)),
        "pr": nx.pagerank(G, weight=weight),
    }
    if weight == "capacity":
        res["betw"] = nx.betweenness_centrality(G, weight="rec_capacity", k=None)
    else:
        res["betw"] = nx.betweenness_centrality(G, weight=weight, k=None)
    #if weight == None:
    #    res["harm"] = nx.harmonic_centrality(G)
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
    
def get_corr_sequence(stats, corr_type, centralities, adjacent=True):
    res = dict([(c,[]) for c in centralities])
    for idx in range(1,len(stats)):
        if adjacent:
            df1 = stats[idx-1]
        else:
            df1 = stats[0]
        df2 = stats[idx]
        merged_df = df1.merge(df2, on="index", suffixes=("_0","_1"))
        merged_df = merged_df.fillna(0.0)
        for cent in  centralities:
            res[cent].append(calc_corr(merged_df, cent, corr_type))
    return res
    
def show_corr_time_series(stats, adjacent, weight_cols, centralities, corr_methods=["pearson","spearman","kendall","w_kendall"]):
    results = {}
    for w in weight_cols:
        results[w] = dict([(corr, get_corr_sequence(stats[w], corr, adjacent=adjacent, centralities=centralities)) for corr in corr_methods])
    for cent in  centralities:
        f, axis = plt.subplots(1, len(weight_cols), sharey=True, figsize=(20,5))
        for i, w in enumerate(weight_cols):
            for corr in corr_methods:
                vals = results[w][corr][cent]
                axis[i].set_title("%s: weight=%s" % (cent, w))
                axis[i].plot(range(1,len(vals)+1), vals, label=corr)
                axis[i].legend()
    return results

def regroup_by_weights(stats, centralities, snapshot_ids, weight_cols):
    regrouped = {}
    for cent in centralities:
        regrouped[cent] = []
        for idx in snapshot_ids:
            mx = stats[weight_cols[0]][idx][["index",cent]]
            for w in weight_cols[1:]:
                mx_tmp = stats[w][idx][["index",cent]]
                mx = mx.merge(mx_tmp, on="index", suffixes=("","_%s" % w))
            mx.columns = ["index"] + weight_cols
            regrouped[cent].append(mx)
    return regrouped

def get_mean_cross_correlation(stats, snapshot_ids, key, method="spearman"):
    mx = corr_mx(stats[key][0].set_index("index"), method=method)
    for idx in snapshot_ids[1:]:
        mx = np.add(mx, corr_mx(stats[key][idx].set_index("index"), method=method))
    return mx / len(snapshot_ids)

def show_mean_cross_correlations(stats, snapshot_ids, key_cols, corr_methods, vmin=0.2, vmax=1.0, sharey=False):
    cross_corr_vals = {}
    for corr in corr_methods:
        cross_corr_vals[corr] = {}
        f, axis = plt.subplots(1, len(key_cols), figsize=(20,5), sharey=sharey)
        for i, key in enumerate(key_cols):
            axis[i].set_title("%s: %s" % (corr, key))
            vals = get_mean_cross_correlation(stats, snapshot_ids, key, method=corr)
            cross_corr_vals[corr][key] = vals
            sns.heatmap(vals, annot=True, vmax=vmax, vmin=vmin, cmap="coolwarm", ax=axis[i])
    return cross_corr_vals
    
### node attachments ###

def get_new_node_attachements(snapshot_idx, snapshot_graphs, snapshot_edges, centrality_ranks, weight_cols):
    known_nodes = set(snapshot_graphs[snapshot_idx-1].nodes())
    print(len(known_nodes))
    edges = snapshot_edges[snapshot_idx]
    attachments_df = edges[~edges["src"].isin(known_nodes)]
    for w in weight_cols:
        attachments_df = attachments_df.merge(centrality_ranks[w][snapshot_idx], left_on="trg", right_on="node_pub", how="left", suffixes=('', '_%s' % w))
    print(attachments_df.shape)
    return attachments_df.rename({"src":"new_node","trg":"old_node"}, axis=1)

def observe_node_attachements_over_time(snapshot_ids, snapshot_graphs, snapshot_edges, centrality_ranks, weight_cols):
    attachments = []
    for idx in snapshot_ids[1:]:
        # discover new node attachements
        attachments_df = get_new_node_attachements(idx, snapshot_graphs, snapshot_edges, centrality_ranks, weight_cols)
        attachments.append(attachments_df)
    return attachments

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
    return pd.DataFrame(corr, columns=df.columns, index=df.columns)

def pop_corr_with_centralities(pop_df, centrality_scores, weight_cols, method="spearman"):
    pop_df_2 = pop_df.reset_index()
    corrs = []
    for i in range(pop_df.shape[1]):
        df = centrality_scores[weight_cols[0]][i].merge(pop_df_2[["index",i]], on="index", how="left")
        df = df.fillna(0.0)
        for w in weight_cols[1:]:
            df = df.merge(centrality_scores[w][i], on="index", suffixes=('', '_%s' % w))
        df = df.drop("index", axis=1)
        cols = list(df.columns)
        cols.remove(i)
        corrs.append(dict(corr_mx(df, method=method)[i][cols]))
    return pd.DataFrame(corrs)

def plot_corr_time_series_with_pop(sp, ke, wk, col_keys):
    x = range(sp.shape[0])
    f, axis = plt.subplots(1, 3, sharey=True, figsize=(20,5))
    for k in col_keys:
        axis[0].set_title("spearman")
        axis[0].plot(x, sp[k], label="%s" % k)
    for k in col_keys:
        axis[1].set_title("wkendall")
        axis[1].plot(x, wk[k], label="%s" % k)
    for k in col_keys:
        axis[2].set_title("kendall")
        axis[2].plot(x, ke[k], label="%s" % k)
    plt.legend()