#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("/home/huabai/scripts")
import numpy as np
from utils.memory_utils import check_mem
from utils.os_utils import create_dir
import os
import re
from glob import glob
import rocksdb

def get_hash_entry_dtype():
    out_dtype = np.dtype([("key",      "u8"),
                          ("chi_idx",  "u2")])
    return out_dtype

if __name__ == "__main__":
    pwd = os.getcwd()
    pwd = re.sub("/mnt", "", pwd)
    pwd = re.sub("/laobai/digs", "", pwd)

    in_dir  = f"{pwd}/DBs_local"
    out_dir = in_dir
    create_dir(out_dir)

    dat_list = glob(f"{in_dir}/final_*.dat")
    print(dat_list)

    hash_dtype = get_hash_entry_dtype()
    id_pattern = re.compile("final_(.+).dat")

    for in_dat in dat_list:
        print("------------")
        print(in_dat)
        hash_type = id_pattern.findall(in_dat)[0]
        in_array = np.memmap(in_dat, mode="r+", dtype=hash_dtype)

        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.max_open_files = 300000
        opts.write_buffer_size = 67108864
        opts.max_write_buffer_number = 3
        opts.target_file_size_base = 67108864

        opts.table_factory = rocksdb.BlockBasedTableFactory(
            filter_policy=rocksdb.BloomFilterPolicy(10),
            block_cache=rocksdb.LRUCache(2 * (1024 ** 3)),
            block_cache_compressed=rocksdb.LRUCache(500 * (1024 ** 2)))

        db = rocksdb.DB(f"{out_dir}/{hash_type}.db", opts)

        current_key = in_array["key"][0]
        current_list = []
        for i in range(in_array.shape[0]):
            if i >0 and not in_array["key"][i] == current_key:
                current_key_bytes = current_key.tobytes()
                current_array = np.array(current_list)
                current_array_bytes = current_array.tobytes()
                # print(current_array_bytes)
                db.put(current_key_bytes, current_array_bytes)
                current_key = in_array["key"][i]
                current_list = []

            current_list.append(in_array["chi_idx"][i])
            if i % 1000000 == 0:
                print(i)
                check_mem()
