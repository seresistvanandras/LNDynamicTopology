import json, time, os
import pandas as pd
import numpy as np
import networkx as nx
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

### preprocessing ###

def extract_snapshot_ends(data_dir, start_time, day_threshold):
    files =  [f for f in sorted(os.listdir(data_dir)) if ".json" in f]
    idx = 0
    snapshot_ends = [start_time]
    # TODO: why the first element is left out?
    for f in files[1:]:
        t = int(f.split(".")[0])
        diff = (t-snapshot_ends[idx])//86400
        if diff >= day_threshold:
            snapshot_ends.append(t)
            idx += 1
    return snapshot_ends

def extract_additional_dates(previous_day, last_day, day_threshold):
    date_ids = []
    delta = 86400*day_threshold
    for ts in range(previous_day,last_day-delta,delta):
        date_ids.append(time.strftime('%m%d',  time.gmtime(ts+delta)))
    return date_ids

def load_temp_data(json_files, node_keys=["pub_key","last_update"], edge_keys=["node1_pub","node2_pub","last_update","capacity"]):
    """Load LN graph json files from several snapshots"""
    node_info, edge_info = [], []
    for idx, json_f in enumerate(json_files):
        with open(json_f) as f:
            tmp_json = json.load(f)
        new_nodes = pd.DataFrame(tmp_json["nodes"])[node_keys]
        new_edges = pd.DataFrame(tmp_json["edges"])[edge_keys]
        new_nodes["snapshot_id"] = idx
        new_edges["snapshot_id"] = idx
        print(json_f, len(new_nodes), len(new_edges))
        node_info.append(new_nodes)
        edge_info.append(new_edges)
    edges = pd.concat(edge_info)
    edges["capacity"] = edges["capacity"].astype("int64")
    edges["last_update"] = edges["last_update"].astype("int64")
    print("All edges:", len(edges))
    edges_no_loops = edges[edges["node1_pub"] != edges["node2_pub"]]
    print("All edges without loops:", len(edges_no_loops))
    return pd.concat(node_info), edges_no_loops

def generate_directed_graph(edges, policy_keys=['disabled', 'fee_base_msat', 'fee_rate_milli_msat', 'min_htlc']):
    directed_edges = []
    indices = edges.index
    #for _, row in edges.iterrows():
    for idx in tqdm(indices):
        row = edges.loc[idx]
        e1 = [row[x] for x in ["snapshot_id","node1_pub","node2_pub","last_update","channel_id","capacity"]]
        e2 = [row[x] for x in ["snapshot_id","node2_pub","node1_pub","last_update","channel_id","capacity"]]
        if row["node2_policy"] == None:
            e1 += [None for x in policy_keys]
        else:
            e1 += [row["node2_policy"][x] for x in policy_keys]
        if row["node1_policy"] == None:
            e2 += [None for x in policy_keys]
        else:
            e2 += [row["node1_policy"][x] for x in policy_keys]
        directed_edges += [e1, e2]
    cols = ["snapshot_id","src","trg","last_update","channel_id","capacity"] + policy_keys
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

def get_snapshots(edges_df, weight_cols=None):
    """Split the LN network edges into snapshots based on the provided time window"""
    cols = None if weight_cols==None else weight_cols.copy()
    snapshot_graphs = []
    snapshot_edges = []
    snapshot_ids = sorted(edges_df["snapshot_id"].unique())
    for i in snapshot_ids:
        snap_edges = edges_df[edges_df["snapshot_id"] == i]
        # drop channels without capacity (if any exists)
        snap_edges = snap_edges[snap_edges["capacity"]>0]
        snapshot_edges.append(snap_edges)
        if weight_cols != None and "capacity" in weight_cols:
            # reciprocal capacity for betweeness centrality computation
            snap_edges["rec_capacity"] = 1.0 / snap_edges["capacity"]
            cols.append("rec_capacity")
        if weight_cols != None and "num_channels" in weight_cols:
            snap_edges["rec_num_channels"] = 1.0 / snap_edges["num_channels"]
            cols.append("rec_num_channels")
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
    elif weight == "num_channels":
        res["betw"] = nx.betweenness_centrality(G, weight="rec_num_channels", k=None)
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
    """'ties' parameter only works for spearman correlation"""
    s = df.shape[1]
    arr = df.values
    corr = np.eye(s)
    for i in range(s):
        for j in range(i+1,s):
            if method == "wkendall":
                corr[i,j] = st.weightedtau(arr[:,i],arr[:,j])[0]
            elif method == "kendall":
                corr[i,j] = st.kendalltau(arr[:,i],arr[:,j])[0]
            elif method == "pearson":
                corr[i,j] = st.pearsonr(arr[:,i],arr[:,j])[0]
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