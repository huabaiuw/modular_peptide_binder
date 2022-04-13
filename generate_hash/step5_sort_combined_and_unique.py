#!/usr/bin/env python
import sys
sys.path.append("/home/huabai/scripts")
import numpy as np
from utils.memory_utils import check_mem
import os
import re
from glob import glob

def get_hash_entry_dtype():
    out_dtype = np.dtype([("key",      "u8"),
                          ("chi_idx",  "u2")])
    return out_dtype

if __name__ == "__main__":

    pwd = os.getcwd()
    pwd = re.sub("/mnt", "", pwd)
    pwd = re.sub("/laobai/digs", "", pwd)
    scratch_dir = "/home/huabai/net_scratch/DBs_scratch"

    in_dir = f"{pwd}/DBs_local"
    out_dir = in_dir
    hash_dtype = get_hash_entry_dtype()

    dat_list = glob(f"{in_dir}/combined*.dat")
    print(dat_list)

    id_pattern = re.compile("combined_(.+).dat")
    for in_dat in dat_list:
        print(in_dat)
        hash_type = id_pattern.findall(in_dat)[0]

        in_array = np.memmap(in_dat, mode="r+", dtype=hash_dtype)
        in_array.sort(order="key")

        print(in_array.shape)
        print(in_array[4])
        print(np.array_equal(in_array[4], in_array[14]))

        zero_array = np.zeros(1, dtype=hash_dtype)
        if np.array_equal(in_array[0], zero_array[0]):
            print(zero_array)

        out_dat = f"{in_dir}/final_{hash_type}.dat"
        out_shape = in_array.shape
        out_array = np.memmap(out_dat, mode="w+", dtype=hash_dtype, shape=out_shape)

        chunk_size = 1000000
        chunk_count = in_array.shape[0] // chunk_size + 1
        print(in_array.shape)
        print(chunk_size)
        print(chunk_count)

        out_array_N = 0
        for i in range(chunk_count):
            if i == chunk_count - 1:
                temp_array = np.unique(in_array[chunk_size*i:in_array.shape[0]])
            else:
                temp_array = np.unique(in_array[chunk_size*i:chunk_size*(i+1)])

            if i > 0 and np.array_equal(temp_array[0], out_array[out_array_N-1]):
                print("duplicate")
                temp_array = temp_array[1:]

            temp_shape = temp_array.shape
            out_array[out_array_N: out_array_N + temp_shape[0]] = temp_array[:]

            out_array_N += temp_shape[0]

            print(out_array_N)
            check_mem()
        out_array.base.resize(out_array_N*hash_dtype.itemsize)
