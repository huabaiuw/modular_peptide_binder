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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import scipy.stats
from pymol import cmd,util,stored

pwd = os.getcwd()

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


exp_csv  = f"{pwd}/ratio_df_Lib01_PXX13_exp.csv"
sort_csv = f"{pwd}/ratio_df_Lib02_PXX13_10nM_LRP6.csv"
id_pattern = re.compile("ratio_df_Lib[0-9]+_([0-9A-Z]+)_.+.csv")

base_id = id_pattern.findall(exp_csv)[0]
WT_DNA_str = WT_dict[base_id]["DNA"]
WT_AA_str  = WT_dict[base_id]["AA"]

## merge the dataframe
exp_df  = pd.read_csv(exp_csv)
sort_df = pd.read_csv(sort_csv)
print(sort_df)

merge_df = exp_df.merge(sort_df, left_on="full_AA", right_on="full_AA", suffixes=("_exp", "_sort"))

## calculate the enrichment ratio of sort group and expression group
merge_df["enrichment"] = merge_df["Ratio_sort"] / merge_df["Ratio_exp"]
# WT_enrichment = merge_df[merge_df["mut_list_sort"].map(lambda x: len(eval(x)) == 0)]["enrichment"].values[0]
## no WT sequence in sort group
## so not normalized to the value of WT enrichment
merge_df["log_norm_enrichment"] = np.log10(merge_df["enrichment"])

## rank the dataframe
ranked_df = merge_df.sort_values(by="log_norm_enrichment", ascending=False)
ranked_df.reset_index(inplace=True)
ranked_df = ranked_df[["mut_list_exp", "log_norm_enrichment", "full_AA"]]
print(ranked_df)






## create a dataframe for the mutation matrix
# unit_len = int( len(WT_AA_str) / 6 )

exp_core_AA = exp_df.loc[0, "AA"]
exp_full_AA = exp_df.loc[0, "full_AA"]

start_idx = exp_full_AA.index(exp_core_AA)
end_idx   = start_idx + len(exp_core_AA)

WT_list = [f"{WT_AA_str[ii]}{ii + 1}" for ii in range(start_idx, end_idx)]
mutation_matrix = pd.DataFrame(index=list(IUPAC.IUPACProtein.letters), columns=WT_list)
mutation_matrix = mutation_matrix.fillna(0)
print(mutation_matrix)

for ii in range(ranked_df.shape[0]):
    mut_id = eval( ranked_df.loc[ii, "mut_list_exp"] )[0]
    mut_pattern = re.compile("([A-Z])([0-9]+)([A-Z])")
    mut_id_list = mut_pattern.findall(mut_id)[0]
    enrichment = ranked_df.loc[ii, "log_norm_enrichment"]
    column_index = f"{mut_id_list[0]}{mut_id_list[1]}"
    row_index    = f"{mut_id_list[2]}"
    current_value = mutation_matrix.loc[row_index, column_index]
    new_value = current_value + enrichment
    mutation_matrix.loc[row_index, column_index] = new_value


exp_mut_list = []
for x in exp_df["mut_list"]:
    exp_mut_list.extend(eval(x))

exp_mut_list = sorted(exp_mut_list)
## create an array to store the WT AA, and label the heatmap
annot_list = []
for ii, mut_char in enumerate( list(IUPAC.IUPACProtein.letters) ):
    sub_list = []
    # print(f"----------{mut_char}------------")
    for jj in range(start_idx, end_idx):
        WT_char = WT_AA_str[jj]
        this_mut = f"{WT_char}{jj+1}{mut_char}"

        if WT_char == mut_char:
            sub_list.append(WT_char)
        elif this_mut in exp_mut_list:
            sub_list.append("")
        else:
            sub_list.append("*")

    annot_list.append(sub_list)
annot_array = np.array(annot_list)


max_value = mutation_matrix.to_numpy().max()
min_value = mutation_matrix.to_numpy().min()

red_blue_ratio = np.abs(max_value - 0) / np.abs(min_value - 0)
red_bins = 128
blue_bins = int(128 / red_blue_ratio) + 1

## turn missing value in sort df to min_value, totally depeleted.
sort_mut_list = []
for x in sort_df["mut_list"]:
    sort_mut_list.extend(eval(x))

sort_mut_list = sorted(sort_mut_list)

for ii, mut_char in enumerate( list(IUPAC.IUPACProtein.letters) ):
    sub_list = []
    # print(f"----------{mut_char}------------")
    for jj in range(start_idx, end_idx):
        WT_char = WT_AA_str[jj]
        this_mut = f"{WT_char}{jj+1}{mut_char}"

        if WT_char == mut_char:
            continue
        elif this_mut in sort_mut_list:
            continue
        else:
            column_index = f"{WT_char}{jj+1}"
            row_index    = f"{mut_char}"
            mutation_matrix.loc[row_index, column_index] = min_value


## create a new cmap
top = cm.get_cmap('Reds', 128)
top_array = top( np.linspace(0, 1, red_bins) )
bottom = cm.get_cmap('Blues_r', 128)
bottom_array = bottom( np.linspace(0, 1, blue_bins))

newcolors = np.vstack( (bottom_array, top_array) )
newcmp = ListedColormap(newcolors, name="for_SSM")

fig, ax = plt.subplots(figsize=(15,7))

sns.heatmap(mutation_matrix, annot=annot_array, fmt="", ax=ax, cmap=newcmp, vmin=min_value, vmax=max_value)

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

plt.savefig(f"{base_id}_SSM_heatmap.png", dpi=300)










#
#
#   ## ==================== color the pdb using pymol ================================ ##
#   ## use pymol to show the entropy as color spectrum
#
#   input_pdb = f"{pwd}/input_pdbs/chainA_R6{base_id}.pdb"
#
#   max_dict = defaultdict(float)
#   for ii in range(unit_len*2, unit_len*3):
#       column_id = f"{WT_AA_str[ii]}{ii+1}"
#       max_dict[ii+1] = mutation_matrix[column_id].max()
#
#   min_dict = defaultdict(float)
#   for ii in range(unit_len*2, unit_len*3):
#       column_id = f"{WT_AA_str[ii]}{ii+1}"
#       min_dict[ii+1] = mutation_matrix[column_id].min()
#
#   ## for max_dict, enriched potential
#   cmd.reinitialize()
#   cmd.load(input_pdb, object=base_id)
#   cmd.bg_color("white")
#   cmd.color("green")
#
#   ## store the resi for each atom in a list
#   stored.resi = []
#   cmd.iterate(base_id, "stored.resi.append(resi)")
#
#   ## store the entropy value for each atom
#   stored.entropy = [max_dict[int(x)] for x in stored.resi]
#
#   spec_min = min(stored.entropy)
#   spec_max = max(stored.entropy)
#
#   ## hack the b factor of the pose
#   cmd.alter(base_id, "b=stored.entropy.pop(0)")
#
#   ## color the pose with b value as customized spectrum
#   colors = ["white", "red"]
#   colvec = [cmd.get_color_tuple(i) for i in colors]
#   parts = len(colvec) - 1
#   expression  = "b"
#   minmax_expr = "b"
#   discrete_expr = ["index", "resi"]
#   selection = f"resi {unit_len*2+1}-{unit_len*3}"
#   cmd.count_atoms(selection)
#
#   quiet = 1
#   if None in [spec_min, spec_max]:
#       stored.e = list()
#       cmd.iterate(selection, 'stored.e.append(%s)' % (minmax_expr))
#       if spec_min is None:
#           spec_min = min(stored.e)
#       if spec_max is None:
#           spec_max = max(stored.e)
#   spec_min, spec_max = float(spec_min), float(spec_max)
#   if not quiet:
#       print(' Spectrum: range (%.5f to %.5f)' % (spec_min, spec_max))
#
#   if spec_max == spec_min:
#       print('no spectrum possible, only equal %s values' % (expression))
#
#   if expression in discrete_expr:
#       val_range = int(spec_max - spec_min + 1)
#   else:
#       val_range = spec_max - spec_min
#       cmd.color(colors[0], selection)
#
#   steps = int( 60 / parts )
#   steps_total = steps * parts
#
#   val_start = spec_min
#   for p in range(parts):
#       for i in range(steps):
#           ii = float(i) / steps
#           col_list = [colvec[p + 1][j] * ii + colvec[p][j] * (1.0 - ii) for j in range(3)]
#           col_name = '0x%02x%02x%02x' % (int( col_list[0] * 255 ), int( col_list[1] * 255 ), int(col_list[2] * 255))
#           val_end = val_range * (i + 1 + p * steps) / steps_total + spec_min
#           if expression in discrete_expr:
#               cmd.color(col_name, '(%s) and %s %d-%d' % (selection, expression, val_start, val_end))
#           else:
#               cmd.color(col_name, '(%s) and %s > %f' % (selection, expression, val_start))
#           val_start = val_end
#
#   cmd.show("sticks", selection)
#   util.cnc()
#   cmd.hide("(h.)")
#
#   cmd.set_view ('''
#       -0.120775372,    0.257371426,   -0.958731830,\
#        0.417765111,   -0.862932444,   -0.284280390,\
#       -0.900489032,   -0.434862316,   -0.003299284,\
#        0.000000000,    0.000000000, -201.924072266,\
#        3.569387436,    1.436904907,   -8.498337746,\
#      159.198486328,  244.649658203,  -20.000000000 ''')
#
#   cmd.save(f"enriched_{base_id}.pse")
#
#   cmd.set("ray_opaque_background", "1")
#   # cmd.bg_color("black")
#   cmd.ray(1200, 800)
#   cmd.png(f"enriched_{base_id}.png", dpi=600)
#
#
#   ## for min_dict, depleted potential
#   cmd.reinitialize()
#   cmd.load(input_pdb, object=base_id)
#   cmd.bg_color("white")
#   cmd.color("green")
#
#   ## store the resi for each atom in a list
#   stored.resi = []
#   cmd.iterate(base_id, "stored.resi.append(resi)")
#
#   ## store the entropy value for each atom
#   stored.entropy = [min_dict[int(x)] for x in stored.resi]
#
#   spec_min = min(stored.entropy)
#   spec_max = max(stored.entropy)
#
#   ## hack the b factor of the pose
#   cmd.alter(base_id, "b=stored.entropy.pop(0)")
#
#   ## color the pose with b value as customized spectrum
#   colors = ["blue", "white"]
#   colvec = [cmd.get_color_tuple(i) for i in colors]
#   parts = len(colvec) - 1
#   expression  = "b"
#   minmax_expr = "b"
#   discrete_expr = ["index", "resi"]
#   selection = f"resi {unit_len*2+1}-{unit_len*3}"
#   cmd.count_atoms(selection)
#
#   quiet = 1
#   if None in [spec_min, spec_max]:
#       stored.e = list()
#       cmd.iterate(selection, 'stored.e.append(%s)' % (minmax_expr))
#       if spec_min is None:
#           spec_min = min(stored.e)
#       if spec_max is None:
#           spec_max = max(stored.e)
#   spec_min, spec_max = float(spec_min), float(spec_max)
#   if not quiet:
#       print(' Spectrum: range (%.5f to %.5f)' % (spec_min, spec_max))
#
#   if spec_max == spec_min:
#       print('no spectrum possible, only equal %s values' % (expression))
#
#   if expression in discrete_expr:
#       val_range = int(spec_max - spec_min + 1)
#   else:
#       val_range = spec_max - spec_min
#       cmd.color(colors[0], selection)
#
#   steps = int( 60 / parts )
#   steps_total = steps * parts
#
#   val_start = spec_min
#   for p in range(parts):
#       for i in range(steps):
#           ii = float(i) / steps
#           col_list = [colvec[p + 1][j] * ii + colvec[p][j] * (1.0 - ii) for j in range(3)]
#           col_name = '0x%02x%02x%02x' % (int( col_list[0] * 255 ), int( col_list[1] * 255 ), int(col_list[2] * 255))
#           val_end = val_range * (i + 1 + p * steps) / steps_total + spec_min
#           if expression in discrete_expr:
#               cmd.color(col_name, '(%s) and %s %d-%d' % (selection, expression, val_start, val_end))
#           else:
#               cmd.color(col_name, '(%s) and %s > %f' % (selection, expression, val_start))
#           val_start = val_end
#
#   cmd.show("sticks", selection)
#   util.cnc()
#   cmd.hide("(h.)")
#
#   cmd.set_view ('''
#       -0.120775372,    0.257371426,   -0.958731830,\
#        0.417765111,   -0.862932444,   -0.284280390,\
#       -0.900489032,   -0.434862316,   -0.003299284,\
#        0.000000000,    0.000000000, -201.924072266,\
#        3.569387436,    1.436904907,   -8.498337746,\
#      159.198486328,  244.649658203,  -20.000000000 ''')
#
#   cmd.save(f"depleted_{base_id}.pse")
#
#   cmd.set("ray_opaque_background", "1")
#   # cmd.bg_color("black")
#   cmd.ray(1200, 800)
#   cmd.png(f"depleted_{base_id}.png", dpi=600)
