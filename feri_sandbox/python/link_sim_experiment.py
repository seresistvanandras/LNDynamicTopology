import pandas as pd
import numpy as np
import networkx as nx
import os

from link_prediction_simulator import LinkSimulator

### experiment ###

def load_graph_snapshots(file_path, ts_upper_bound=1553390858, verbose=True):
    snapshots = pd.read_csv(file_path)
    snapshots = snapshots[snapshots["last_update"] < ts_upper_bound]
    time_boundaries = list(snapshots.groupby("snapshot_id")["last_update"].max())
    for i in range(len(time_boundaries)-1):
        diff = time_boundaries[i+1]-time_boundaries[i]
        if verbose:
            print(diff)
        if diff <= 0:
            print("Exiting in snapshot %i due to negative time boundary difference!" % i)
            time_boundaries = time_boundaries[:(i+1)]
            break
    return snapshots, time_boundaries

def process_links_for_simulator(top_k, links_df, time_boundaries, verbose=True):
    links_tmp = links_df.copy()
    links_tmp["record_id"] = links_tmp.index
    links_tmp = links_tmp[["src","trg","capacity","time","eval","record_id"]]
    links_tmp["top_k"] = top_k
    # set snapshot id for the simulation: 
    link_snapshots = np.digitize(list(links_tmp["time"]), time_boundaries)
    # simulate always on the previous graph snapshot
    links_tmp["snapshot"] = link_snapshots - 1
    if verbose:
        print("ALL:", links_tmp["snapshot"].value_counts())
        print("EVAL:", links_tmp[links_tmp["eval"]==1]["snapshot"].value_counts())
    return links_tmp

class SimulatedLinkPredExperiment():
    def __init__(self, top_k, half_life, snapshot_file_path, links_file_path, node_meta_file_path, tx_fee_sat, tx_num, eps, drop_disabled=True, drop_low_cap=True, with_depletion=True, time_window=None, only_triangles=False, verbose=False):
        self.verbose = verbose
        self.top_k = top_k
        self.half_life = half_life
        self.only_triangles = only_triangles
        # simulation parameters
        self.tx_fee_sat = tx_fee_sat
        self.tx_num = tx_num
        self.eps = eps
        self.drop_disabled = drop_disabled
        self.drop_low_cap = drop_low_cap
        self.with_depletion = with_depletion
        self.time_window = time_window
        # load files
        self.snapshots, self.time_boundaries = load_graph_snapshots(snapshot_file_path, verbose=self.verbose)
        # node labels
        node_meta = pd.read_csv(node_meta_file_path)
        self.providers = list(node_meta["pub_key"])
        # prediction records
        self.links_df = pd.read_csv(links_file_path)
        # model id
        self.experiment_id = "%isat_k%i_e%.2f_dd%s_dlc%s_wd%s_tw%s_ot%s_hl%s" % (tx_fee_sat, tx_num, eps, drop_disabled, drop_low_cap, with_depletion, str(time_window), only_triangles, str(half_life))
    
    def preprocess(self):
        self.links_for_sim = process_links_for_simulator(self.top_k, self.links_df, self.time_boundaries, verbose=True)
        
    def run(self, output_prefix, snapshots_ids=None, max_threads=4):
        output_dir = "%s/%s" % (output_prefix, self.experiment_id)
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if snapshots_ids == None:
            snapshots_ids = sorted(list(self.links_for_sim["snapshot"].unique()))
            if -1 in snapshots_ids:
                snapshots_ids.remove(-1)
        for snap_id in snapshots_ids:
            print(snap_id)
            snap_edges = self.snapshots[self.snapshots["snapshot_id"] == snap_id]
            G = nx.from_pandas_edgelist(snap_edges, source="src", target="trg", edge_attr=["capacity"], create_using=nx.MultiDiGraph())
            sim = LinkSimulator(snap_edges, self.providers, self.tx_fee_sat, self.tx_num, eps=self.eps, drop_disabled=self.drop_disabled, drop_low_cap=self.drop_low_cap, with_depletion=self.with_depletion, time_window=self.time_window)
            ranks, preds = sim.predict(G, self.links_for_sim, snap_id, self.top_k, self.only_triangles, self.half_life)
            header = ["record_id", "src", "trg", "time", "capacity", "eval", "snapshot", "global_traffic", "global_income", "inbound_depletions", "high_degree", "high_cap"]
            pred_cols = header[-5:]
            ranks_df = pd.DataFrame(list(ranks), columns=header)
            ranks_df.to_csv("%s/snapshot_%i.csv" % (output_dir, snap_id), index=False)
            predictions = dict()
            for col in pred_cols:
                predictions[col] = []
            for event_preds in preds:
                for idx, col in enumerate(pred_cols):
                    predictions[col].append(event_preds[idx])
            for col in pred_cols:
                pd.concat(predictions[col], ignore_index=True).to_csv("%s/preds_%s_%i.csv" % (output_dir, col, snap_id), index=False)