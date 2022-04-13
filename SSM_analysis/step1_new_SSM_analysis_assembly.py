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

# lib_name_dict = { 1:  ("PXX13_exp",         "Lib01"),
#                   2:  ("PXX13_10nM_LRP6",   "Lib02"),
#                   5:  ("PXX28_exp",         "Lib05"),
#                   6:  ("PXX28_1nM_PAW6",    "Lib06"),
#                   7:  ("PXX28_1nM_PEW6",    "Lib07"),
#                   8:  ("PXX28_1nM_PKW6",    "Lib08"),
#                   9:  ("PXX28_1nM_PRW6",    "Lib09"),
#                   16: ("R8D2_exp",          "Lib16"),
#                   17: ("R8D2_100nM_PPL6",   "Lib17"),
#                   20: ("R15_exp",           "Lib20"),
#                   21: ("R15_1uM_IYP6",      "Lib21") }


# lib_name_dict = {
#                   1:  ("PXX13_exp",         "Lib01"),
#                   5:  ("PXX28_exp",         "Lib05"),
#                   9:  ("PXX28_1nM_PRW6",    "Lib09"),
#                 }


lib_name_dict = {
                  10:  ("PXX28_1nM_bioPEW6_1nM_PKW6",         "Lib10"),
                  11:  ("PXX28_1nM_bioPEW6_3nM_PKW6",         "Lib11"),
                  12:  ("PXX28_1nM_bioPEW6_10nM_PKW6",        "Lib12"),
                }

fastq_dir = f"{pwd}/fastq"

## step4: make file summary
## collect all file ids and names
gz_list = glob(f"{fastq_dir}/*.gz")

## example name: DHR-SSM-Dec2020-8_S20_L002_R1_001.fastq.gz
id_pattern = re.compile("DHR-SSM-Dec2020-([0-9]+)_S([0-9]+)_L([0-9]+)_R([12]+)_.+fastq.gz")
FASTQ_files = defaultdict(lambda:defaultdict(lambda:defaultdict(str)))

for each_gz in gz_list:
    # print(each_gz)
    id_info = id_pattern.findall(each_gz)
    if not id_info:
        print("no id match")
        continue

    lib_num = int(id_info[0][0])
    S_num   = int(id_info[0][1])
    L_num   = int(id_info[0][2])
    R_num   = int(id_info[0][3])

    if not lib_num in lib_name_dict.keys(): continue

    lib_name = f"{lib_name_dict[lib_num][1]}_{lib_name_dict[lib_num][0]}"
    print(lib_name)

    if R_num == 1:
        FASTQ_files[lib_name][L_num]["F"] = each_gz
    elif R_num == 2:
        FASTQ_files[lib_name][L_num]["RC"] = each_gz
    else:
        continue

## step5: go through all libraries:
print( FASTQ_files.keys() )
for lib_name in FASTQ_files.keys():
    print(f"================ working on lib {lib_name} ===================")

    ## extract the WT sequence information
    base_id = lib_name.split("_")[1]
    print(base_id)

    WT_DNA_str = WT_dict[base_id]["DNA"]
    adaptor_5prime = adaptors_dict[base_id]["5prime"]
    adaptor_3prime = adaptors_dict[base_id]["3prime"]

    print(WT_DNA_str)
    print(adaptor_5prime)
    print(adaptor_3prime)

    ## the core is the sequence range ordered on CHIP
    core_pattern = re.compile(f"(.+){adaptor_5prime}(.+){adaptor_3prime}")
    core_DNA_str = core_pattern.findall(str(WT_DNA_str))[0][1]
    core_DNA_seq = Seq(core_DNA_str, IUPAC.unambiguous_dna)
    core_AA_seq = core_DNA_seq.translate()

    ## the padded_seq is the sequence preceding the CHIP order
    padded_DNA_str = core_pattern.findall(str(WT_DNA_str))[0][0] + adaptor_5prime
    padded_DNA_seq = Seq(padded_DNA_str, IUPAC.unambiguous_dna)
    padded_AA_seq = padded_DNA_seq.translate()
    padded_len = len(padded_AA_seq)

    AAseq_counter = defaultdict(int)
    ## each library will have a couple of channels, the L_num
    for L_num in FASTQ_files[lib_name].keys():
        fastq_pair = FASTQ_files[lib_name][L_num]
        print(fastq_pair)

        ## A_dict is the R1 sequences, the forward sequencing
        A_dict = {}
        A_pattern = re.compile(f"{adaptor_5prime}(.+)")

        with gzip.open(fastq_pair["F"], "rt") as gz_f:
            for record in SeqIO.parse(gz_f, "fastq"):
                try:
                    DNA_str_A = A_pattern.findall(str(record.seq)) [0]
                except Exception:
                    continue

                if "N" in DNA_str_A:
                    continue

                if not len(DNA_str_A) % 3 == 0:
                    DNA_str_A = DNA_str_A[:-(len(DNA_str_A) % 3)]
                assert len(DNA_str_A) % 3 == 0
                DNA_seq_A = Seq(DNA_str_A, IUPAC.unambiguous_dna)
                AA_seq_A = DNA_seq_A.translate()

                asterisk_match_list = []
                for asterisk_match in re.finditer("\*", str(AA_seq_A)):
                    asterisk_index = asterisk_match.span()[0]
                    assert AA_seq_A[asterisk_match.span()[0]:asterisk_match.span()[1]] == "*"
                    asterisk_match_list.append(asterisk_index)

                if len(AA_seq_A) == 0:
                    continue

                if len(asterisk_match_list) > 1:
                    continue

                elif len(asterisk_match_list) == 1 and asterisk_match_list[0] != (len(AA_seq_A)-1):
                    continue

                elif len(asterisk_match_list) == 1 and asterisk_match_list[0] == (len(AA_seq_A)-1):
                    A_dict[record.id] = AA_seq_A[:-1]
                else:
                    A_dict[record.id] = AA_seq_A

        ## B_dict is the R2 sequences, the reverse complement sequencing
        B_dict = {}
        B_pattern = re.compile(f"(.+){adaptor_3prime}")

        with gzip.open(fastq_pair["RC"], "rt") as gz_f:
            for record in SeqIO.parse(gz_f, "fastq"):
                try:
                    DNA_str_B = B_pattern.findall(str(record.seq.reverse_complement())) [0]
                except Exception:
                    continue

                if "N" in DNA_str_B:
                    continue

                DNA_len = len(DNA_str_B)

                DNA_str_B = DNA_str_B[DNA_len%3:]
                assert len(DNA_str_B) % 3 == 0

                DNA_seq_B = Seq(DNA_str_B, IUPAC.unambiguous_dna)

                AA_seq_B = DNA_seq_B.translate()

                asterisk_match_list = []
                for asterisk_match in re.finditer("\*", str(AA_seq_B)):
                    asterisk_index = asterisk_match.span()[0]
                    assert AA_seq_B[asterisk_match.span()[0]:asterisk_match.span()[1]] == "*"
                    asterisk_match_list.append(asterisk_index)

                if len(AA_seq_B) == 0:
                    continue

                if len(asterisk_match_list) > 1:
                    continue
                elif len(asterisk_match_list) == 1 and asterisk_match_list[0] != 0:
                    continue
                elif len(asterisk_match_list) == 1 and asterisk_match_list[0] == 0:
                    B_dict[record.id] = AA_seq_B[1:]
                else:
                    B_dict[record.id] = AA_seq_B

        ## compare with WT sequence
        ## and make a spliced sequence
        WT_seq_len = len(core_AA_seq)
        for each_key in A_dict.keys():
            try:
                A_AA = str( A_dict[each_key] )
                B_AA = str( B_dict[each_key] )
            except Exception:
                continue

            A_len = len(A_AA)
            B_len = len(B_AA)

            ## if there is no overlap of the two parts
            if A_len + B_len <= WT_seq_len:
                continue
                # spliced_AA = A_AA + core_AA_seq[A_len: WT_seq_len-B_len] + B_AA

            ## if there is overlap of the two parts
            else:
                overlap_len = A_len + B_len - WT_seq_len

                if A_AA[-overlap_len:] != B_AA[:overlap_len]: continue

                spliced_AA = A_AA[:-overlap_len] + B_AA
            try:
                assert len(spliced_AA) == WT_seq_len
            except Exception:
                continue

            AAseq_counter[spliced_AA] += 1

            if debug:
                print( A_AA )
                print( f"{B_AA:>{WT_seq_len}}" )
                print(spliced_AA)
                print(core_AA_seq)
                print("-"*WT_seq_len)

    AAseq_df = pd.DataFrame( AAseq_counter.items(), columns=['AA', "count"] )
    AAseq_df.sort_values(by=['count'], ascending=False, inplace=True, ignore_index=True)
    out_csv_name = f"result_df_{lib_name}.csv"
    AAseq_df.to_csv(out_csv_name, index=False)
