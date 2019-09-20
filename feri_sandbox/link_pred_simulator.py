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
            cost, router_fees, _ = process_path(p, row["amount_SAT"], capacity_map, G,  "total_fee")
            if src in p[1:-1]:
                src_idx = p.index(src)
                src_revenue = router_fees[src]
                next_ = p[src_idx+1].replace("_trg","")
                if next_ in targets:
                    target_effect[next_] += src_revenue
            else:
                continue
        except nx.NetworkXNoPath:
            continue
        except:
            print(idx)
            print(p)
            raise
    prediction_df = pd.DataFrame()
    prediction_df["item"] = targets
    prediction_df["opt_income"] = [target_effect.get(trg, 0.0) for trg in targets]
    return prediction_df

def get_possible_targets(G, source_node):
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
    degrees = G.degree(nbunch=possible_targets)
    capacities = G.degree(nbunch=possible_targets, weight="capacity")
    target_nodes = list(pd.DataFrame(degrees, columns=["node","degree"]).sort_values("degree", ascending=False)["node"])
    return target_nodes, degrees, capacities

def augment_capacity_map(init_capacities, G, transactions, valid_cap, source_node, target_nodes, cap, amount_sat):
    capacity_map = copy.deepcopy(init_capacities)
    G_tmp = G.copy()
    # create possible targets
    tx_targets = set(transactions["target"])
    default_fee = calculate_tx_fee(pd.Series({"fee_base_msat":1000.0,"fee_rate_milli_msat":1.0}), amount_sat)
    new_edges, final_targets = [], []
    for trg in target_nodes:
        if (source_node,trg) in capacity_map or (trg, source_node) in capacity_map:
            continue
        else:
            final_targets.append(trg)
            if valid_cap:
                is_trg_prov = trg in tx_targets
                # update capacity map
                capacity_map[(source_node,trg)] = [cap, default_fee, is_trg_prov, cap]            
                capacity_map[(trg,source_node)] = [0, default_fee, is_trg_prov, cap]                 # handle additional edges
                new_edges.append((source_node, trg, default_fee))
                if is_trg_prov:
                    new_edges.append((source_node, str(trg)+"_trg", 0.0))
    if valid_cap:
        G_tmp.add_weighted_edges_from(new_edges, weight="total_fee")
    return capacity_map, G_tmp, final_targets

def get_target_ranks(total_depletions, router_traffic, router_income, init_capacities, G, transactions, amount_sat, weight, pred_event_params):
    rec_id, source_node, base_model_predictions, true_target, ts, cap = pred_event_params
    valid_cap = (cap >= amount_sat)
    # extract possible prediction set
    top_k = int(base_model_predictions)
    #top_k = len(base_model_predictions)
    target_nodes, degrees, capacities = get_possible_targets(G, source_node)
    capacity_map, G_tmp, final_targets = augment_capacity_map(init_capacities, G, transactions, valid_cap, source_node,  target_nodes, cap, amount_sat)
    # execute all predictions
    predictions = simulate_target_effects(capacity_map, G_tmp, transactions, source_node, final_targets, weight=weight)
    predictions["inbound_depletions"] = [total_depletions.get(trg, 0) for trg in final_targets]
    predictions["global_traffic"] = [router_traffic.get(trg, 0) for trg in final_targets]
    predictions["global_income"] = [router_income.get(trg, 0.0) for trg in final_targets]
    # fill with meta info
    predictions["record_id"] = rec_id
    predictions["time"] = ts
    predictions["user"] = source_node
    predictions["high_degree"] = predictions["item"].apply(lambda x: degrees[x])
    predictions["high_cap"] = predictions["item"].apply(lambda x: capacities[x])
    pred_cols = ["opt_income","global_traffic","global_income","inbound_depletions","high_degree","high_cap"]
    pred_ranks = predictions[pred_cols].rank(method="first", ascending=False)
    pred_cols = ["rank_"+col for col in pred_cols]
    pred_ranks.columns = pred_cols
    predictions = pd.concat([predictions,pred_ranks], axis=1)
    if true_target in final_targets:
        all_ranks = list(predictions[predictions["item"]==true_target][pred_cols].values[0])
        ranks = [r if r <= top_k else None for r in all_ranks]
    elif true_target in set(G.nodes()):
        ranks = [None] * len(pred_cols)
        print("%s true target is part of existing channel!" % true_target)
        #raise RuntimeError("%s true target is part of existing channel!" % true_target)
    else:
        ranks = [None] * len(pred_cols)
    pred_toplists = []
    for col in pred_cols:
        feat = col.replace("rank_","")
        top_k_preds = predictions[predictions[col]<=top_k][["record_id","time","user","item",feat,col]]
        pred_toplists.append(top_k_preds.rename({col:"rank",feat:"prediction"}, axis=1).sort_values("rank"))
    return ranks, pred_toplists

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
        # preparation
        _, _, router_info_df, total_depletions = get_shortest_paths(self.current_capacity_map, self.G, self.transactions, hash_transactions=False, cost_prefix="", weight=weight)
        router_traffic = dict(router_info_df["node"].value_counts())
        router_income = dict(router_info_df.groupby("node")["fee"].sum())
        print("simulation on original snapshot data is DONE")
        # prediction part
        pred_events_params = list(zip(pred_events["record_id"],pred_events["src"], pred_events["target_node_set"], pred_events["trg"], pred_events["time"], pred_events["capacity"]))
        if max_threads > 1:
            f_partial = functools.partial(get_target_ranks, total_depletions, router_traffic, router_income, self.current_capacity_map, self.G, self.transactions, self.amount_sat, weight)
            executor = concurrent.futures.ProcessPoolExecutor(max_threads)
            ranks = list(executor.map(f_partial, pred_events_params))
            executor.shutdown()
        else:
            ranks = [get_target_ranks(total_depletions, router_traffic, router_income, self.current_capacity_map, self.G, self.transactions, self.amount_sat, weight, pe) for pe in pred_events_params]
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

def process_links_for_simulator(top_k, links_df, preds, time_boundaries, only_eval=True, verbose=False):
    links_tmp = links_df.copy()
    if only_eval:
        links_tmp = links_tmp[links_tmp["eval"]==1]
    links_tmp["record_id"] = links_tmp.index
    # set target node set
    if preds is None:
        links_for_sim = links_tmp[["src","trg","capacity","time","eval","record_id"]]
        links_for_sim["target_node_set"] = top_k
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
    def __init__(self, top_k, snapshot_file_path, links_file_path, preds_file_path, node_meta_file_path, tx_fee_sat, tx_num, eps=0.05, alpha=None, drop_disabled=True):
        self.top_k = top_k
        # simulation parameters
        self.tx_fee_sat = tx_fee_sat
        self.tx_num = tx_num
        self.eps = eps
        self.alpha = alpha
        self.drop_disabled = drop_disabled
        # load files
        self.snapshots, self.time_boundaries = load_graph_snapshots(snapshot_file_path)
        self.links_df = pd.read_csv(links_file_path)
        self.preds = None#pd.read_csv(preds_file_path)
        #self.preds = self.preds.sort_values(["record_id","rank"])
        # model id
        base_model_id = preds_file_path.split("/")[-1].replace("preds_","").replace(".csv","")
        simulator_id = "%isat_k%i_a%s_e%.2f_drop%s" % (tx_fee_sat, tx_num, str(alpha), eps, drop_disabled)
        #self.experiment_id = "trial"
        self.experiment_id = simulator_id + "-" + base_model_id
        # node labels
        node_meta = pd.read_csv(node_meta_file_path)
        self.providers = list(node_meta["pub_key"])
    
    def preprocess(self):
        self.links_for_sim = process_links_for_simulator(self.top_k, self.links_df, self.preds, self.time_boundaries, verbose=True)
        
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
            #link_events = self.links_for_sim[self.links_for_sim["snapshot"]==snap_id].head(40)
            sim = LinkPredSimulator(snap_edges, self.providers, self.tx_fee_sat, self.tx_num, eps=self.eps, alpha=self.alpha, drop_disabled=self.drop_disabled)
            df = link_events
            pred_cols = ["opt_income","global_traffic","global_income","inbound_depletions","high_degree","high_cap"]
            result = sim.simulate(df,max_threads=max_threads)
            ranks, preds = zip(*result)
            ranks_df = pd.DataFrame(list(ranks), columns=pred_cols)
            df = pd.DataFrame(np.concatenate([df.values,ranks_df.values], axis=1), columns=list(df.columns)+pred_cols)
            df.drop("target_node_set", axis=1, inplace=True)
            df[["record_id","src","trg","time","capacity","eval","snapshot"]+pred_cols].to_csv("%s/snapshot_%i.csv" % (output_dir, snap_id), index=False)
            predictions = dict()
            for col in pred_cols:
                predictions[col] = []
            for event_preds in preds:
                for idx, col in enumerate(pred_cols):
                    predictions[col].append(event_preds[idx])
            for col in pred_cols:
                pd.concat(predictions[col], ignore_index=True).to_csv("%s/preds_%s_%i.csv" % (output_dir, col, snap_id), index=False)
        return sim