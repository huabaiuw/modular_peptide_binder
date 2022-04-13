#!/usr/bin/env python
import os
import gzip
from glob import glob
import re
from collections import defaultdict
from random import shuffle

pwd = os.getcwd()
pwd = re.sub("/mnt", "", pwd)
pwd = re.sub("/laobai/digs", "", pwd)
cmd_script = f"{pwd}/step03_generate_hash.py"





##  NQBB_N_1HD2_OD1_4LHD_A_811_832.pdb
# id_pattern = re.compile("key_pair_([A-Z]{2,})_")

input_list = []
for a,b,c in os.walk("./pairs_seeds"):
    for f in c:
        if f.endswith(".pdb"):
            input_list.append(f"{a}/{f}")

cmds_list = []
for each_pdb in input_list:
    cmd_line = f"{cmd_script} {each_pdb}\n"
    cmds_list.append(cmd_line)
shuffle(cmds_list)

# job_type = "backfill"
job_type = "short"
memory_size = "4g"
ncpu = 1


out_file = f"job_list_step3_{job_type}.list"
with open(out_file, "w") as task_cmds_f:
    for each_cmd in cmds_list:
        task_cmds_f.write(each_cmd)

submit_script = f"./submit_step3_to_digs_{job_type}.sbatch"
with open(submit_script, "w") as submit_f:
    submit_f.write("#!/bin/bash\n")
    submit_f.write(f"#SBATCH -p {job_type}\n")
    submit_f.write(f"#SBATCH --mem={memory_size}\n")
    submit_f.write(f"#SBATCH -o /dev/null\n")
    submit_f.write(f"#SBATCH -n {ncpu}\n")
    submit_f.write("#SBATCH -N 1\n")
    submit_f.write("\n")
    # submit_f.write("export OMP_NUM_THREADS=1\n")
    submit_f.write(f"eval `cat {out_file} | head -${{SLURM_ARRAY_TASK_ID}} | tail -1`\n")
    submit_f.write("exit 0\n")

print("============== cmd example to submit prepare fragments jobs: ======================")
print(f"sbatch -a 1-{len(cmds_list)} {submit_script}")
print("===================================================================================")
