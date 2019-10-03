import pandas as pd
from transaction_simulator import TransactionSimulator

def get_target_ranks(link_record, possible_targets, top_k, total_depletions, router_traffic, router_income, degrees, capacities):
    source_node, true_target, cap, ts, rec_id = link_record["src"], link_record["trg"], link_record["capacity"], link_record["time"], link_record["record_id"]
    scores = pd.DataFrame()
    scores["item"] = possible_targets
    # execute all predictions
    scores["global_traffic"] = [router_traffic.get(trg, 0) for trg in possible_targets]
    scores["global_income"] = [router_income.get(trg, 0.0) for trg in possible_targets]
    scores["inbound_depletions"] = [total_depletions.get(trg, 0) for trg in possible_targets]
    scores["high_degree"] = [degrees.get(trg, 0.0) for trg in possible_targets]
    scores["high_cap"] = [capacities.get(trg, 0.0) for trg in possible_targets]
    # fill with meta info
    scores["record_id"] = rec_id
    scores["time"] = ts
    scores["user"] = source_node
    pred_cols = ["global_traffic", "global_income", "inbound_depletions", "high_degree", "high_cap"]
    ranks = scores[pred_cols].rank(method="first", ascending=False)
    pred_cols = ["rank_"+col for col in pred_cols]
    ranks.columns = pred_cols
    predictions = pd.concat([scores, ranks], axis=1)
    if true_target in possible_targets:
        # pick the row of the true target
        model_ranks = list(predictions[predictions["item"]==true_target][pred_cols].values[0])
        # truncate ranks at the boundary value top_k
        adjusted_model_ranks = [r if r <= top_k else None for r in model_ranks]
    else:
        adjusted_model_ranks = [None] * len(pred_cols)
    pred_toplists = []
    for col in pred_cols:
        feat = col.replace("rank_","")
        top_k_preds = predictions[predictions[col] <= top_k][["record_id", "time", "user", "item", feat,col]]
        pred_toplists.append(top_k_preds.rename({col : "rank", feat : "prediction"}, axis=1).sort_values("rank"))
    return adjusted_model_ranks, pred_toplists

def collect_predictions(link_events, snapshot_id, top_k, only_triangles, G, neighbors, router_traffic, router_income, total_depletions):
    eval_event = 0
    src_not_present = 0
    src_not_in_snapshot = 0
    trg_not_present = 0
    trg_not_in_snapshot = 0
    predictions = []
    G_undir = G.to_undirected()
    node_set = set(G.nodes())
    degrees = dict(G.degree())
    capacities = dict(G.degree(weight="capacity"))
    for idx, row in link_events.iterrows():
        src_, trg_,  eval_ = row["src"], row["trg"], row["eval"]
        if eval_ == 1:
            eval_event += 1
            if src_ not in G_undir.nodes():
                src_not_present += 1
            if trg_ not in G.nodes():
                src_not_in_snapshot += 1
            if trg_ not in G_undir.nodes():
                trg_not_present += 1
            if trg_ not in G.nodes():
                trg_not_in_snapshot += 1
            possible_targets = list(node_set.difference(neighbors[src_]))
            if only_triangles and src_ in G_undir.nodes():
                triangle_endpoints = []
                for node in possible_targets:
                    if node in G_undir.nodes() and len(set(G_undir.neighbors(src_)).intersection(G_undir.neighbors(node)))>0:
                        triangle_endpoints.append(node)
                possible_targets = triangle_endpoints
            ranks, preds = get_target_ranks(row, possible_targets, top_k, total_depletions, router_traffic, router_income, degrees, capacities)
            meta = [row["record_id"], src_, trg_, row["time"], row["capacity"], eval_, snapshot_id]
            ranks_with_meta = meta + ranks
            predictions.append((ranks_with_meta, preds))
        if not src_ in neighbors:
            neighbors[src_] = set()
        neighbors[src_].add(trg_)
        G_undir.add_edges_from([(src_,trg_)])
    print("Missing event ratios:")
    for cnt in [src_not_present, trg_not_present, src_not_in_snapshot, trg_not_in_snapshot]:
        print(cnt / eval_event)
    return predictions

def former_neighbors(links, snapshot_id):
    former_links = links[links["snapshot"] < snapshot_id]
    # all edges are mirrored
    neighbors = dict(former_links.groupby("src")["trg"].apply(set))
    return neighbors
        
class LinkSimulator(TransactionSimulator):
    def __init__(self, edges, providers, amount_sat, k, eps=0.8, drop_disabled=True, drop_low_cap=True, with_depletion=True, time_window=None, verbose=True):
        super(LinkSimulator, self).__init__(edges, providers, amount_sat, k, eps, drop_disabled, drop_low_cap, with_depletion, time_window, verbose)
        print(self.params)
    
    def predict(self, G, links_for_sim, snap_id, top_k, only_triangles, weight="total_fee"):
        # preparation
        _, _, router_info_df, total_depletions = self.simulate(weight=weight, with_node_removals=False, max_threads=1, verbose=False)
        router_traffic = dict(router_info_df["node"].value_counts())
        router_income = dict(router_info_df.groupby("node")["fee"].sum())
        print("simulation on original snapshot data is DONE")
        neighbors = former_neighbors(links_for_sim, snap_id)
        # prediction part
        link_events = links_for_sim[links_for_sim["snapshot"] == snap_id]
        results = collect_predictions(link_events, snap_id, top_k, only_triangles, G, neighbors, router_traffic, router_income, total_depletions)
        ranks, preds = zip(*results)
        return ranks, preds