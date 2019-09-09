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
        n1p, n2p, chan_id, last_update, cap = row["src"], row["trg"], row["channel_id"], row["last_update"], row["capacity"]
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
        n1, n2, t, cap = row["n1p"], row["n2p"], row["time"], row["capacity"]
        if row["new_channel"] & row["new_edge"] & row["homophily"]:
            # evaluate a random direction for new homophily edges
            if row["rnd"] < 0.5:
                link_pred_edges.append((n1,n2,t,cap,1))
                link_pred_edges.append((n2,n1,t,cap,0))
            else:
                link_pred_edges.append((n2,n1,t,cap,1))
                link_pred_edges.append((n1,n2,t,cap,0))
        else:
            # no eval for other edges
            link_pred_edges.append((n1,n2,t,cap,0))
            link_pred_edges.append((n2,n1,t,cap,0))
    return pd.DataFrame(link_pred_edges, columns=["src","trg","time","capacity","eval"])

def encode_nodes(links_df):
    nodes = set(links_df["src"]).union(set(links_df["trg"]))
    recoder = dict(zip(nodes,range(len(nodes))))
    links_df["user"] = links_df["src"].apply(lambda x: recoder[x])
    links_df["item"] = links_df["trg"].apply(lambda x: recoder[x])
    return links_df[["src","trg","capacity","user","item","time","eval"]]
    
def prepare_link_prediction_data(events):
    # Random selection of edge direction
    links_df = select_random_direction_for_eval(events)
    return encode_nodes(links_df)

from alpenglow.evaluation import DcgScore

def calculate_dcg_for_preds(preds, links):
    preds['dcg'] = DcgScore(preds)
    hits = preds[~preds["score"].isnull()]
    links_tmp = links[links["eval"]==1].copy().merge(hits, on=["time","user","item"], how="left")
    print("preds:", links_tmp["dcg"].fillna(0.0).mean())  

def get_rankings(model, links, exclude_known, pred_file=None):
    if pred_file == None:
        rankings = model.run(links, exclude_known=exclude_known, verbose=True)
    else:
        rankings = model.run(links, exclude_known=exclude_known, calculate_toplists=links['eval'], verbose=True)
        preds = model.get_predictions()
        links_with_score = links.copy()
        links_with_score["score"] = 1.0
        preds_joined = preds.join(
            links_with_score.reset_index().set_index(['index', 'item'])['score'],
            on=['record_id', 'item'],
            how="left"
        )
        calculate_dcg_for_preds(preds_joined, links)
        preds_joined.to_csv(pred_file, index=False)
    rankings['dcg'] = DcgScore(rankings)
    print(rankings["dcg"].mean())
    return rankings