import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, operator
from transaction_simulator import *

### transaction simulator ###

def simulate_target_effects(G, transactions, src, targets, weight=None):
    target_effect = dict([(val,0.0) for val in targets])
    for idx, row in transactions.iterrows():
        try:
            p = nx.shortest_path(G, source=row["source"], target=row["target"] + "_trg", weight=weight)
            # if src is a router node in the transaction
            if src in p[1:-1]:
                cost, router_fees = process_path(p, row["amount_SAT"], G, "total_fee")
                src_idx = p.index(src)
                src_revenue = router_fees[src]
                prev_, next_ = p[src_idx-1], p[src_idx+1]
                prevb, nextb = prev_ in targets, next_ in targets
                if prevb or nextb:
                    if nextb and prevb:
                        target_effect[prev_] += src_revenue / 2.0
                        target_effect[next_] += src_revenue / 2.0
                    elif prevb:
                        target_effect[prev_] += src_revenue
                    else:
                        target_effect[next_] += src_revenue
                else:
                    continue
            else:
                continue
        except RuntimeError as re:
            raise re
        except:
             continue
    prediction_df = pd.DataFrame()
    # it is important to keep the order of the original predictions
    prediction_df["node"] = targets
    prediction_df["score"] = [target_effect.get(trg, 0.0) for trg in targets]
    return list(prediction_df.sort_values("score", ascending=False)["node"])

def get_target_ranks(G, transactions, amount_sat, weight, pred_event_params):
    source_node, target_nodes, true_target, ts = pred_event_params
    G_tmp = G.copy()
    #target_nodes_tmp = []
    #for trg in target_nodes:
    #    if G_tmp.has_edge(source_node, trg):
    #        print("%i: %s-%s exists" % (ts, source_node, trg))
    #        continue
    #    target_nodes_tmp.append(trg)
    target_nodes_tmp = target_nodes
    prediction = target_nodes_tmp
    if len(target_nodes_tmp) > 0:
        new_edges = pd.DataFrame([])
        new_edges["trg"] = target_nodes_tmp
        new_edges["src"] = source_node
        new_edges["fee_base_msat"] = 1000.0
        new_edges["fee_rate_milli_msat"] = 1.0
        new_edges["total_fee"] = calculate_tx_fee(new_edges, amount_sat)
        tuples = list(zip(new_edges["src"], new_edges["trg"], new_edges["total_fee"]))
        #tuples += list(zip(new_edges["trg"], new_edges["src"], new_edges["total_fee"]))
        #print(G_tmp.number_of_edges())
        G_tmp.add_weighted_edges_from(tuples, weight="total_fee")
        prediction = simulate_target_effects(G_tmp, transactions, source_node, target_nodes, weight=weight)
    sim_based_rank = prediction.index(true_target)+1.0 if true_target in prediction else None
    rank_in_list = target_nodes_tmp.index(true_target)+1.0 if true_target in target_nodes_tmp else None
    return sim_based_rank, rank_in_list

import functools
import concurrent.futures

class LinkPredSimulator():
    def __init__(self, edges, providers, amount_sat, k, eps=0.05, alpha=None, drop_disabled=True):
        self.amount_sat = amount_sat
        self.edges = prepare_edges_for_simulation(edges, amount_sat, drop_disabled)
        self.node_variables, self.providers = init_node_params(self.edges, providers, eps, alpha)
        self.transactions = sample_transactions(self.node_variables, amount_sat, k)
        self.G = generate_graph_for_path_search(self.edges, self.transactions)
        print("%i transaction were generated." % k)
    
    def simulate(self, pred_events, weight="total_fee", max_threads=4):
        pred_events_params = list(zip(pred_events["src"], pred_events["target_node_set"], pred_events["trg"], pred_events["time"]))
        if max_threads > 1:
            f_partial = functools.partial(get_target_ranks, self.G, self.transactions, self.amount_sat, weight)
            executor = concurrent.futures.ProcessPoolExecutor(max_threads)
            ranks = list(executor.map(f_partial, pred_events_params))
            executor.shutdown()
        else:
            ranks = [get_target_ranks(self.G, self.transactions, self.amount_sat, weight, pe) for pe in pred_events_params]
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
        links_for_sim = links_tmp[["src","trg","time","eval","record_id"]].merge(preds_as_lists, on="record_id", how="left")
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
        self.experiment_id = "trial"
        #self.experiment_id = "slink_" + simulator_id + "-" + base_model_id
        #self.experiment_id = "sp_" + simulator_id + "-" + base_model_id
        #self.experiment_id = simulator_id + "-" + base_model_id
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
            #link_events = self.links_for_sim[self.links_for_sim["snapshot"]==snap_id]
            link_events = self.links_for_sim[self.links_for_sim["snapshot"]==snap_id].head(20)
            sim = LinkPredSimulator(snap_edges, self.providers, self.tx_fee_sat, self.tx_num, eps=self.eps, alpha=self.alpha, drop_disabled=self.drop_disabled)
            df = link_events#.head(100)
            ranks = sim.simulate(df,max_threads=max_threads)
            sim_based_rank, rank_in_list = zip(*ranks)
            df["rank"] = sim_based_rank
            df["base"] = rank_in_list
            df.drop("target_node_set", axis=1, inplace=True)
            df.to_csv("%s/snapshot_%i.csv" % (output_dir, snap_id), index=False)
        return sim