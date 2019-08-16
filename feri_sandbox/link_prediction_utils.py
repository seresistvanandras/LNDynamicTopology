import pandas as pd
import numpy as np
from tqdm import tqdm

def process_edges(edges):
    print("Keep only first channel event:")
    print(len(edges))
    edges_tmp = edges.drop_duplicates(subset="channel_id", keep="first")
    print(len(edges_tmp))
    print("Extract new edge / homophily information:")
    channel_state = {}
    channel_events = []
    seen_nodes, seen_edges = set(), set()
    indices = edges_tmp.index
    for idx in tqdm(indices, mininterval=10):
        row = edges_tmp.loc[idx]
        # channel events
        n1p, n2p, chan_id, last_update, cap = row["node1_pub"], row["node2_pub"], row["channel_id"], row["last_update"], row["capacity"]
        is_new_channel = chan_id not in channel_state
        if (n1p,n2p) in seen_edges or (n2p,n1p) in seen_edges:
            is_new_edge = False
        else:
            is_new_edge = True
            seen_edges.add((n1p,n2p))
        if n1p in seen_nodes and n2p in seen_nodes:
            is_homophily = True
        else:
            is_homophily = False
            seen_nodes.add(n1p)
            seen_nodes.add(n2p)
        channel_state[chan_id] = cap
        channel_events.append([n1p, n2p, last_update, chan_id, is_new_channel, is_new_edge, is_homophily, cap])
    channel_events_df = pd.DataFrame(channel_events, columns=["n1p","n2p","time","channel_id","new_channel","new_edge","homophily","capacity"])
    return channel_events_df

def select_random_direction_for_eval(new_channels):
    """We train link prediction model on bi-directional edges BUT evaluate on a single random direction"""
    new_channels["rnd"] = np.random.random(size=len(new_channels))
    link_pred_edges = []
    for idx, row in new_channels.iterrows():
        n1, n2, t = row["n1p"], row["n2p"], row["time"]
        if row["rnd"] < 0.5:
            link_pred_edges.append((n1,n2,t,1))
            link_pred_edges.append((n2,n1,t,0))
        else:
            link_pred_edges.append((n2,n1,t,1))
            link_pred_edges.append((n1,n2,t,0))
    return link_pred_edges

def encode_nodes(links_df):
    nodes = set(links_df["src"]).union(set(links_df["trg"]))
    recoder = dict(zip(nodes,range(len(nodes))))
    links_df["user"] = links_df["src"].apply(lambda x: recoder[x])
    links_df["item"] = links_df["trg"].apply(lambda x: recoder[x])
    return links_df[["src","trg","user","item","time","eval"]]
    
def prepare_link_prediction_data(events):
    # Filter for homophily edges
    new_channels = events[events["new_channel"] & events["new_edge"] & events["homophily"]]
    print(new_channels.shape)
    # Random selection of edge direction
    links_df = pd.DataFrame(select_random_direction_for_eval(new_channels), columns=["src","trg","time","eval"])
    return encode_nodes(links_df)