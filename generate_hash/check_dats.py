#!/usr/bin/env python


import numpy as np
from glob import glob


dat_list = glob("./DBs_local/*.dat")

in_dtype = np.dtype([("key",      "u8"),
                      ("chi_idx",  "u2")])

total_n = 0
for each_dat in dat_list:
    print(each_dat)
    in_array = np.memmap(each_dat, mode="r", dtype=in_dtype)

    print(in_array.shape)
    total_n += in_array.shape[0]


print(total_n)
