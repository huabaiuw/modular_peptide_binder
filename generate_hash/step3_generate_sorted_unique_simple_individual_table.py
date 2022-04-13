#!/usr/bin/env python
import sys
sys.path.append("/home/huabai/scripts")

import numpy as np
import os
from utils.memory_utils import check_mem
from collections import defaultdict
import pickle
from glob import glob
import re


def get_hash_entry_dtype():
    out_dtype = np.dtype([("key",      "u8"),
                          ("chi_idx",  "u2")])

    return out_dtype



def generate_unique(in_dat, out_dat):
    hash_dtype = get_hash_entry_dtype()
    try:
        loaded_array = np.memmap(in_dat, mode="r", dtype=hash_dtype)
    except Exception:
        print("failed to load")
        return None

    final_hash_table = np.unique(loaded_array)
    print(final_hash_table.shape)

    if final_hash_table.shape[0] == 1:
        print(final_hash_table)

    if final_hash_table.shape[0] > 1:
        final_hash_table.sort(order="key")
        print("save unqiue")
        final_array = np.memmap(out_dat, mode="w+", dtype=hash_dtype, shape=final_hash_table.shape)
        final_array[:] = final_hash_table[:]





if __name__ == "__main__":
    pwd = os.getcwd()
    pwd = re.sub("/mnt", "", pwd)
    pwd = re.sub("/laobai/digs", "", pwd)
    # scratch_dir = "/home/huabai/net_scratch/DBs_scratch"

    DBs_dir = f"{pwd}/DBs_local"

    in_dir  = DBs_dir

    out_dir = f"{pwd}/DBs_unique"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    input_list = sys.argv[1]

    with open(input_list, "r") as f:
        hash_list = [x.strip() for x in f.readlines() if len(x)>0]

    print(hash_list)
    id_pattern = re.compile("raw_key_chi_idx_(.+).dat")

    for i, each_raw in enumerate(hash_list):
        raw_id = id_pattern.findall(each_raw)[0]
        print(raw_id)
        unique_id = f"unique_{raw_id}"
        print(unique_id)
        out_dat = f"{out_dir}/{unique_id}.dat"
        generate_unique(each_raw, out_dat)
