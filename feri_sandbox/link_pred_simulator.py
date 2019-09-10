import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, operator
from transaction_simulator import *

### transaction simulator ###

def simulate_target_effects(capacity_map, G, transactions, src, targets, weight=None):
    target_effect = dict([(val,0.0) for val in targets])
    for idx, row in transactions.iterrows():
        try:
            p = nx.shortest_path(G, source=row["source"], target=row["target"] + "_trg", weight=weight)
            # if src is a router node in the transaction
            cost, router_fees = process_path(p, row["amount_SAT"], capacity_map, G, "total_fee")
            if src in p[1:-1]:
                src_idx = p.index(src)
                src_revenue = router_fees[src]
                next_ = p[src_idx+1].replace("_trg","")
                if next_ in targets:
                    target_effect[next_] += src_revenue
                #prev_, next_ = p[src_idx-1], p[src_idx+1]
                #prevb, nextb = prev_ in targets, next_ in targets
                #if prevb or nextb:
                #    if nextb and prevb:
                #        target_effect[prev_] += src_revenue / 2.0
                #        target_effect[next_] += src_revenue / 2.0
                #    elif prevb:
                #        target_effect[prev_] += src_revenue
                #    else:
                #        target_effect[next_] += src_revenue
                #else:
                #    continue
            else:
                continue
        except RuntimeError as re:
            raise re
        except:
             continue
    in_degs = dict(G.in_degree(nbunch=targets))
    prediction_df = pd.DataFrame()
    # it is important to keep the order of the original predictions
    prediction_df["item"] = targets
    prediction_df["prediction"] = [target_effect.get(trg, 0.0) for trg in targets]
    prediction_df["in_degree"] = [in_degs[trg] for trg in targets]
    #return list(prediction_df.sort_values("prediction", ascending=False)["item"])
    return prediction_df.sort_values("prediction", ascending=False)

def get_target_ranks(init_capacities, G, transactions, amount_sat, weight, pred_event_params):
    rec_id, source_node, target_nodes, true_target, ts, cap = pred_event_params
    # extract possible prediction set
    top_k = len(target_nodes)
    NN = set(G.nodes())
    N = NN.copy()
    for n in NN:
        if "_trg" in n:
            N.remove(n)
    if source_node in N:
        neighbors = set(G.to_undirected(as_view=True).neighbors(source_node))
        possible_targets = N.difference(neighbors)
        possible_targets.remove(source_node)
    else:
        possible_targets = N
    possible_degrees = G.degree(nbunch=possible_targets)
    possible_capacities = G.degree(nbunch=possible_targets, weight="capacity")
    target_nodes = list(pd.DataFrame(possible_degrees, columns=["node","degree"]).sort_values("degree", ascending=False)["node"])
    with_top_caps = list(pd.DataFrame(possible_capacities, columns=["node","degree"]).sort_values("degree", ascending=False)["node"])[:top_k]
    # do NOT simulate with capacity below amount!
    if cap >= amount_sat:
        capacity_map = copy.deepcopy(init_capacities)
        G_tmp = G.copy()
        tx_targets = set(transactions["target"])
        default_fee = calculate_tx_fee(pd.Series({"fee_base_msat":1000.0,"fee_rate_milli_msat":1.0}), amount_sat)
        new_edges, final_targets = [], []
        for trg in possible_targets:
            if (source_node,trg) in capacity_map or (trg, source_node) in capacity_map:
                continue
            else:
                final_targets.append(trg)
                is_trg_prov = trg in tx_targets
                # update capacity map
                capacity_map[(source_node,trg)] = [cap, default_fee, is_trg_prov, cap]
                capacity_map[(trg,source_node)] = [0, default_fee, is_trg_prov, cap]
                # handle additional edges
                new_edges.append((source_node, trg, default_fee))
                if is_trg_prov:
                    new_edges.append((source_node, str(trg)+"_", 0.0))
        G_tmp.add_weighted_edges_from(new_edges, weight="total_fee")
        prediction_with_scores = simulate_target_effects(capacity_map, G_tmp, transactions, source_node, final_targets, weight=weight)
        full_cap_nodes = list(prediction_with_scores.sort_values("in_degree",ascending=True)["item"])[:top_k]
        prediction_with_scores = prediction_with_scores.head(top_k)
    else:
        prediction_with_scores = pd.DataFrame()
        prediction_with_scores["item"] = target_nodes[:top_k]
        prediction_with_scores["prediction"] = None
        in_degs = G.in_degree(nbunch=possible_targets)
        full_cap_nodes = list(pd.DataFrame(in_degs, columns=["node","degree"]).sort_values("degree", ascending=True)["node"])[:top_k]
    prediction_with_scores["record_id"] = rec_id
    prediction_with_scores["time"] = ts
    prediction_with_scores["user"] = source_node
    prediction_with_scores["rank"] = np.array(range(len(prediction_with_scores))) + 1.0
    prediction = list(prediction_with_scores["item"])
    target_nodes = target_nodes[:top_k]
    rank1 = prediction.index(true_target)+1.0 if true_target in prediction else None
    rank2 = target_nodes.index(true_target)+1.0 if true_target in target_nodes else None
    rank3 = with_top_caps.index(true_target)+1.0 if true_target in with_top_caps else None
    rank4 = full_cap_nodes.index(true_target)+1.0 if true_target in full_cap_nodes else None
    return rank1, rank2, rank3, rank4, prediction_with_scores[["record_id","user","item","rank","prediction"]]

import functools
import concurrent.futures

class LinkPredSimulator():
    def __init__(self, edges, providers, amount_sat, k, eps=0.05, alpha=None, drop_disabled=True):
        self.amount_sat = amount_sat
        self.edges = prepare_edges_for_simulation(edges, amount_sat, drop_disabled)
        self.node_variables, self.providers = init_node_params(self.edges, providers, eps, alpha)
        self.transactions = sample_transactions(self.node_variables, amount_sat, k)
        self.current_capacity_map, self.edges_with_capacity = init_capacities(self.edges, set(self.transactions["target"]), amount_sat)
        self.G = generate_graph_for_path_search(self.edges_with_capacity, self.transactions)
        print("%i transactions were generated." % k)
    
    def simulate(self, pred_events, weight="total_fee", max_threads=4):
        pred_events_params = list(zip(pred_events["record_id"],pred_events["src"], pred_events["target_node_set"], pred_events["trg"], pred_events["time"], pred_events["capacity"]))
        if max_threads > 1:
            f_partial = functools.partial(get_target_ranks, self.current_capacity_map, self.G, self.transactions, self.amount_sat, weight)
            executor = concurrent.futures.ProcessPoolExecutor(max_threads)
            ranks = list(executor.map(f_partial, pred_events_params))
            executor.shutdown()
        else:
            ranks = [get_target_ranks(self.current_capacity_map, self.G, self.transactions, self.amount_sat, weight, pe) for pe in pred_events_params]
        return ranks
    
### experiment ###

def load_graph_snapshots(file_path, ts_upper_bound=1553390858, verbose=False):
    snapshots = pd.read_csv(file_path)
    # Remove records with outlier timestamp (above 2019-08-16)
    snapshots = snapshots[snapshots["last_update"] < ts_upper_bound]
    time_boundaries = list(snapshots.groupby("snapshot_id")["last_update"].max())
    for i in range(len(time_boundaries)-1):
        diff = time_boundaries[i+1]-time_boundaries[i]
        if verbose:
            print(diff)
        if diff <= 0:
            print("Exiting in snapshot %i due to negative time boundary difference!" % i)
            break
    time_boundaries = time_boundaries[:(i+1)]
    return snapshots, time_boundaries

def process_links_for_simulator(links_df, preds, time_boundaries, only_eval=True, verbose=False):
    links_tmp = links_df.copy()
    if only_eval:
        links_tmp = links_tmp[links_tmp["eval"]==1]
    links_tmp["record_id"] = links_tmp.index
    # set target node set
    if preds is None:
        links_for_sim = links_tmp[["src","trg","time","eval","record_id"]]
    else:
        id_to_pub = dict(zip(links_tmp["user"],links_tmp["src"]))
        id_to_pub.update(dict(zip(links_tmp["item"],links_tmp["trg"])))
        print("Number of nodes:", len(id_to_pub))
        preds["trg_pub"] = preds["item"].replace(id_to_pub)
        preds_as_lists = preds.groupby("record_id")["trg_pub"].apply(list).reset_index()
        preds_as_lists.columns = ["record_id","target_node_set"]
        links_for_sim = links_tmp[["src","trg","capacity","time","eval","record_id"]].merge(preds_as_lists, on="record_id", how="left")
    # set time snapshot
    link_snapshots = np.digitize(list(links_for_sim["time"]), time_boundaries)
    links_for_sim["snapshot"] = link_snapshots - 1 # simulate always on the previous graph snapshot
    links_for_sim = links_for_sim[links_for_sim["snapshot"]>=0] # there is no graph history for the first snapshot
    if verbose:
        print(links_for_sim["snapshot"].value_counts())
    return links_for_sim

class SimulatedLinkPredExperiment():
    def __init__(self, snapshot_file_path, links_file_path, preds_file_path, node_meta_file_path, tx_fee_sat, tx_num, eps=0.05, alpha=None, drop_disabled=True):
        # simulation parameters
        self.tx_fee_sat = tx_fee_sat
        self.tx_num = tx_num
        self.eps = eps
        self.alpha = alpha
        self.drop_disabled = drop_disabled
        # load files
        self.snapshots, self.time_boundaries = load_graph_snapshots(snapshot_file_path)
        self.links_df = pd.read_csv(links_file_path)
        self.preds = pd.read_csv(preds_file_path)
        self.preds = self.preds.sort_values(["record_id","rank"])
        # model id
        base_model_id = preds_file_path.split("/")[-1].replace("preds_","").replace(".csv","")
        simulator_id = "%isat_k%i_a%s_e%.2f_drop%s" % (tx_fee_sat, tx_num, str(alpha), eps, drop_disabled)
        #self.experiment_id = "trial"
        self.experiment_id = simulator_id + "-" + base_model_id
        # node labels
        node_meta = pd.read_csv(node_meta_file_path)
        self.providers = list(node_meta["pub_key"])
    
    def preprocess(self):
        self.links_for_sim = process_links_for_simulator(self.links_df, self.preds, self.time_boundaries, verbose=True)
        
    def run(self, output_prefix, snapshots_ids=None, max_threads=4):
        output_dir = "%s/%s" % (output_prefix, self.experiment_id)
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if snapshots_ids == None:
            snapshots_ids = sorted(list(self.links_for_sim["snapshot"].unique()))
        for snap_id in snapshots_ids:
            print(snap_id)
            snap_edges = self.snapshots[self.snapshots["snapshot_id"]==snap_id]
            link_events = self.links_for_sim[self.links_for_sim["snapshot"]==snap_id]
            #link_events = self.links_for_sim[self.links_for_sim["snapshot"]==snap_id].head(50)
            sim = LinkPredSimulator(snap_edges, self.providers, self.tx_fee_sat, self.tx_num, eps=self.eps, alpha=self.alpha, drop_disabled=self.drop_disabled)
            df = link_events
            ranks = sim.simulate(df,max_threads=max_threads)
            rank1, rank2, rank3, rank4, prediction_parts = zip(*ranks)
            df["rank1"] = rank1
            df["rank2"] = rank2
            df["rank3"] = rank3
            df["rank4"] = rank4
            df.drop("target_node_set", axis=1, inplace=True)
            df.to_csv("%s/snapshot_%i.csv" % (output_dir, snap_id), index=False)
            predictions = pd.concat(prediction_parts, sort=False)
            predictions.to_csv("%s/preds_%i.csv" % (output_dir, snap_id), index=False)
        return sim