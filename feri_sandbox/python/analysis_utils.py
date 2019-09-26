import sys, os
sys.path.insert(0,"../")
from ln_utils import corr_mx

import pandas as pd
import matplotlib.pyplot as plt

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
    """Calculate average cross correlation of independent simulator experiments with the same parameters"""
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