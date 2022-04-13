#!/usr/bin/env python

import sys
sys.path.append("/home/huabai/scripts")
import numpy as np
import os
from utils.memory_utils import check_mem
from collections import defaultdict
import pickle
import re
from glob import glob


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
    scratch_dir = "/home/huabai/net_scratch/DBs_scratch"


    in_dir  = f"{pwd}/DBs_local"
    out_dir = in_dir

    in_list = glob(f"{in_dir}/raw_key_*.dat")

    ## example name: raw_key_chi_idx_NQBB_Q_8533608.dat
    id_pattern = re.compile("raw_key_chi_idx_(NQBB_[A-Z]+)_([0-9]+).dat")
    hash_dict = defaultdict(list)
    for each_dat in in_list:
        print(each_dat)
        hash_type, dat_id = id_pattern.findall(each_dat)[0]
        dat_id = int(dat_id)
        print(hash_type)
        print(dat_id)

        hash_dict[hash_type].append(each_dat)

    for hash_type, hash_list in hash_dict.items():
        print(hash_type)
        hash_dtype = get_hash_entry_dtype()

        combined_hash = f"{out_dir}/combined_{hash_type}.dat"
        grand_N = 0
        for i, each_raw in enumerate(hash_list):
            raw_array = np.memmap(each_raw, mode="r", dtype=hash_dtype)
            unique_array = np.unique(raw_array)
            unique_array.sort(order="key")

            zero_array = np.zeros(1, dtype=hash_dtype)
            if np.array_equal(unique_array[0], zero_array[0]):
                print("!!")
                print(unique_array[0])
                unique_array = unique_array[1:]

            if i == 0:
                combined_array = np.memmap(combined_hash, mode="w+", dtype=hash_dtype, shape=unique_array.shape)
                combined_array[:] = unique_array[:]
                grand_N += unique_array.shape[0]

            else:
                adding_memmap_array = np.memmap(combined_hash, mode="r+", dtype=hash_dtype, shape=unique_array.shape, offset = grand_N*hash_dtype.itemsize)
                adding_memmap_array[:] = unique_array[:]
                grand_N += unique_array.shape[0]
