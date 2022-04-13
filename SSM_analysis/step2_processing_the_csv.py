#!/usr/bin/env python

from Bio import SeqIO
import os
import sys
import argparse
from glob import glob
import subprocess
from multiprocessing import Pool
import time
import pandas as pd
from collections import defaultdict
import re
import gzip
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import numpy as np


pwd = os.getcwd()
debug = False

## step1: get adaptor information
adaptors_fasta = f"{pwd}/NGS_primer_overlaps.fasta"
adaptors_dict = defaultdict(lambda: defaultdict(str))
for each_record in SeqIO.parse(adaptors_fasta, "fasta"):
    base_id, primer_type = each_record.id.split("_")

    if each_record.id.endswith("_F"):
        adaptors_dict[base_id]["5prime"] = str(each_record.seq)[-16:]
    elif each_record.id.endswith("_RC"):
        adaptors_dict[base_id]["3prime"] = str(each_record.seq.reverse_complement())[:16]
    else:
        continue

print(adaptors_dict)

## step2: get WT sequence information
WT_fasta = f"{pwd}/WT_DNA.fasta"
WT_dict = defaultdict(lambda: defaultdict(str))
id_pattern = re.compile("R6(.+)_WT")
for each_record in SeqIO.parse(WT_fasta, "fasta"):
    base_id = id_pattern.findall(each_record.id)[0]
    if not base_id in adaptors_dict.keys():
        continue

    assert adaptors_dict[base_id]["5prime"] in str(each_record.seq)
    assert adaptors_dict[base_id]["3prime"] in str(each_record.seq)

    WT_dict[base_id]["DNA"] = str( each_record.seq )
    WT_dict[base_id]["AA"]  = str( each_record.seq.translate() )
print(WT_dict)

## step3: NGS libraries information
## key: number of the NCS pools
## value: (Lib name, Lib ID)

lib_name_dict = { 1:  ("PXX13_exp",         "Lib01"),
                  2:  ("PXX13_10nM_LRP6",   "Lib02"),
                  5:  ("PXX28_exp",         "Lib05"),
                  6:  ("PXX28_1nM_PAW6",    "Lib06"),
                  7:  ("PXX28_1nM_PEW6",    "Lib07"),
                  8:  ("PXX28_1nM_PKW6",    "Lib08"),
                  9:  ("PXX28_1nM_PRW6",    "Lib09"),
                  16: ("R8D2_exp",          "Lib16"),
                  17: ("R8D2_100nM_PPL6",   "Lib17"),
                  20: ("R15_exp",           "Lib20"),
                  21: ("R15_1uM_IYP6",      "Lib21") }



lib_name_dict = {
                  10:  ("PXX28_1nM_bioPEW6_1nM_PKW6",         "Lib10"),
                  11:  ("PXX28_1nM_bioPEW6_3nM_PKW6",         "Lib11"),
                  12:  ("PXX28_1nM_bioPEW6_10nM_PKW6",        "Lib12"),
                }



def parse_two_AA(AA_A, AA_B):
    assert len(AA_A) == len(AA_B)

    mut_list = []
    for ii, (char_A, char_B) in enumerate(zip(AA_A, AA_B)):
        if char_A != char_B:
            mut_list.append(f"{char_A}{ii+1}{char_B}")

    return mut_list

# csv_list = glob(f"{pwd}/result_df_*.csv")


csv_list = glob(f"{pwd}/result_df_*bioPEW6*.csv")


id_pattern = re.compile("result_df_Lib[0-9]+_([0-9A-Z]+)_.+.csv")
lib_name_pattern = re.compile("result_df_(.+).csv")

for each_csv in csv_list:
    lib_name = lib_name_pattern.findall(each_csv)[0]
    base_id = id_pattern.findall(each_csv)[0]
    WT_DNA_str = WT_dict[base_id]["DNA"]
    WT_AA_str  = WT_dict[base_id]["AA"]

    adaptor_5prime = adaptors_dict[base_id]["5prime"]
    adaptor_3prime = adaptors_dict[base_id]["3prime"]

    ## the core is the sequence range ordered on CHIP
    core_pattern = re.compile(f"(.+{adaptor_5prime})(.+)({adaptor_3prime}.+)")
    core_DNA_str = core_pattern.findall(str(WT_DNA_str))[0][1]

    ## the padded 5prime seq is the sequence preceding the CHIP order
    padded_5prime_DNA_str = core_pattern.findall(str(WT_DNA_str))[0][0]
    assert len(padded_5prime_DNA_str) % 3 == 0
    padded_5prime_DNA_seq = Seq(padded_5prime_DNA_str, IUPAC.unambiguous_dna)
    padded_5prime_AA_seq = padded_5prime_DNA_seq.translate()

    ## the padded 3prime seq is the sequence after the CHIP order
    padded_3prime_DNA_str = core_pattern.findall(str(WT_DNA_str))[0][2]
    assert len(padded_3prime_DNA_str) % 3 == 0
    padded_3prime_DNA_seq = Seq(padded_3prime_DNA_str, IUPAC.unambiguous_dna)
    padded_3prime_AA_seq = padded_3prime_DNA_seq.translate()

    each_df = pd.read_csv(each_csv)
    each_WT_AA = WT_dict[base_id]["AA"]
    each_df["full_AA"] = each_df["AA"].map(lambda x: "".join(padded_5prime_AA_seq + x + padded_3prime_AA_seq) )

    print(WT_AA_str)
    print(each_df["full_AA"][0])

    each_df["mut_list"] = each_df["full_AA"].map(lambda x: parse_two_AA(WT_AA_str, x) )

    # clean_df = each_df[ each_df["mut_list"].apply(lambda x: len(x) < 5) ]
    clean_df = each_df[ each_df["mut_list"].apply(lambda x: len(x) == 1) ]
    clean_df = clean_df[ clean_df["count"] > 1 ]

    clean_df["Ratio"] = clean_df["count"].apply(lambda x: x/sum(clean_df["count"]))

    out_csv_name = f"ratio_df_{lib_name}.csv"
    clean_df.to_csv(out_csv_name, index=False)
