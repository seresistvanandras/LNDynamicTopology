import sys, os, json
sys.path.insert(0,"../")
from ln_utils import corr_mx

import pandas as pd
import matplotlib.pyplot as plt

def relevant_routers():
    # nodes wit at least 50SAT mean daily income + 10 man daily traffic
    return [
        "02ad6fb8d693dc1e4569bcedefadf5f72a931ae027dc0f0c544b34c1c6f3b9a02b",#rompert.com
        "0232e20e7b68b9b673fb25f48322b151a93186bffe4550045040673797ceca43cf",#zigzag.io
        "03e50492eab4107a773141bb419e107bda3de3d55652e6e1a41225f06a0bbf2d56",#yalls.org
        "0279c22ed7a068d10dc1a38ae66d2d6461e269226c60258c021b1ddcdfe4b00bc4",#ln1.satoshilabs.com
        "03abf6f44c355dec0d5aa155bdbdd6e0c8fefe318eff402de65c6eb2e1be55dc3e",#OpenNode
        "03c2abfa93eacec04721c019644584424aab2ba4dff3ac9bdab4e9c97007491dda",#tippin.me
        "0331f80652fb840239df8dc99205792bba2e559a05469915804c08420230e23c7c",#LightningPowerUsers.com
        "03021c5f5f57322740e4ee6936452add19dc7ea7ccf90635f95119ab82a62ae268",#bluewallet - 03021c5f5f57322740e4
        "028dcc199be86786818c8c32bffe9db8855c5fca98951eec99d1fa335d841605c2",#btc.lnetwork.tokyo
        "03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f",#ACINQ
        "0217890e3aad8d35bc054f43acc00084b25229ecff0ab68debd82883ad65ee8266",#1ML.com node ALPHA
        "03a503d8e30f2ff407096d235b5db63b4fcf3f89a653acb6f43d3fc492a7674019",#Sagittarius A
        "02cdf83ef8e45908b1092125d25c68dcec7751ca8d39f557775cd842e5bc127469",#tady je slushovo
        "03bb88ccc444534da7b5b64b4f7b15e1eccb18e102db0e400d4b9cfe93763aa26d",#LightningTo.Me
        "03ee180e8ee07f1f9c9987d98b5d5decf6bad7d058bdd8be3ad97c8e0dd2cdc7ba",#Electrophorus [W_C_B]
        "028303182c9885da93b3b25c9621d22cf34475e63c123942e402ab530c0556e675",#ORANGESQUIRREL
        "03cb7983dc247f9f81a0fa2dfa3ce1c255365f7279c8dd143e086ca333df10e278",#fairly.cheap
        "030c3f19d742ca294a55c00376b3b355c3c90d61c6b6b39554dbc7ac19b141c14f",#Bitrefill.com
        "031678745383bd273b4c3dbefc8ffbf4847d85c2f62d3407c0c980430b3257c403",#lightning-roulette.com
        "0242a4ae0c5bef18048fbecf995094b74bfb0f7391418d71ed394784373f41e4f3",#CoinGate
        "02529db69fd2ebd3126fb66fafa234fc3544477a23d509fe93ed229bb0e92e4fb8",#Boltening.club
        "02df5ffe895c778e10f7742a6c5b8a0cefbe9465df58b92fadeb883752c8107c8f",#Blockstream Store
    ]

def get_experiment_files(experiment_id, snapshots, simulation_dir):
    """Collect simulation resources for multiple snapshots and independent experiments"""
    files = {}
    for snap_id in snapshots:
        files[snap_id] = []
        for f in os.listdir("%s/%i" % (simulation_dir, snap_id)):
            if experiment_id in f:
                files[snap_id].append("%s/%i/%s" % (simulation_dir, snap_id, f))
    return files

def load_data(experiment_files, snapshots, file_postfix):
    """Load the output files of transaction simulator"""
    parts = []
    for snap_id in snapshots:
        data_files = []
        for f in experiment_files[snap_id]:
            sample_id = f.split(".")[-1]
            if file_postfix == "lengths_distrib":
                df = pd.read_csv("%s/%s.csv" % (f, file_postfix), names=["length","count"])
            elif file_postfix == "global_failure_ratios":
                with open("%s/global_failure_ratios.json" % f) as json_f:
                    ratios = json.load(json_f)
                df = pd.DataFrame(ratios.items(), columns=["entity","failure_ratio"])
            else:
                df = pd.read_csv("%s/%s.csv" % (f, file_postfix))
            df["sample"] = int(sample_id)
            data_files.append(df)
        tmp_df = pd.concat(data_files, axis=0, sort=False)
        tmp_df["snapshot_id"] = snap_id
        parts.append(tmp_df)
        #print(snap_id)
    return parts

def avg_cross_corr(df, snapshot_id, col, methods, key_col="node"):
    """Calculate average cross correlation of independent simulator experiments for a given snapshot"""
    snap = df[snapshot_id]
    sample_num = snap["sample"].max()+1
    cols = [key_col, col]
    merged = snap[snap["sample"]==0][cols].rename({col:col+"_0"}, axis=1)
    for i in range(1,sample_num):
        s = snap[snap["sample"]==i][cols].rename({col:col+"_%i" % i}, axis=1)
        merged = merged.merge(s, on=key_col, how="outer").fillna(0.0)
    merged.drop(key_col, inplace=True, axis=1)
    res = {}
    for method in methods:
        cnt = corr_mx(merged, method=method).sum().sum() - sample_num 
        denom = sample_num**2-sample_num
        res[method] = cnt / denom
    return res

def reshape_cross_corr_df(df, methods):
    """Reshape results for seaborn visualiztion"""
    parts = []
    for corr in methods:
        part = df.reset_index()[["index",corr]].copy()
        part["correlation type"] = corr
        parts.append(part.rename({corr:"value"},axis=1))
    return pd.concat(parts, sort=False)