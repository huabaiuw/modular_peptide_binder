#!/usr/bin/env python
import os
import gzip
from glob import glob
import re


pwd = os.getcwd()
pwd = re.sub("/mnt", "", pwd)
pwd = re.sub("/laobai/digs", "", pwd)

scripts_dir = pwd
cmd_script = scripts_dir + "/step01_rr_prototype.py"

out_dir = f"{pwd}/pairs_seeds"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

input_list = "/home/huabai/playground/devel_hash/00_update_pdb_database/cullpdb_pc70.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_14_chains21418"

with open(input_list, "r") as f:
    pdb_list = [x.strip().split()[0] for x in f.readlines()[1:]]

total_job_n = len(pdb_list)
print(total_job_n)
batch_n = 5

total_job_n % batch_n

cmds_list = []

for i in range(batch_n):
    if i < total_job_n % batch_n:
        start_line = (total_job_n // batch_n + 1) * i
        end_line = (total_job_n // batch_n + 1) * (i + 1)
    else:
        start_line = (total_job_n // batch_n) * i + (total_job_n % batch_n)
        end_line = (total_job_n // batch_n) * (i + 1) + (total_job_n % batch_n)
        if end_line > total_job_n: end_line = total_job_n

    cmd_line = f"{cmd_script} {input_list} {start_line} {end_line} {out_dir}\n"
    cmds_list.append(cmd_line)

out_file = "joblist_step01.list"
with open(out_file, "w") as task_cmds_f:
    for each_cmd in cmds_list:
        task_cmds_f.write(each_cmd)

job_type = "short"
memory_size = "1g"
ncpu = 1

submit_script = f"./submit_step01_to_digs_{job_type}.sh"
with open(submit_script, "w") as submit_f:
    submit_f.write("#!/bin/bash\n")
    submit_f.write(f"#SBATCH -p {job_type}\n")
    submit_f.write(f"#SBATCH --mem={memory_size}\n")
    submit_f.write(f"#SBATCH -o /dev/null\n")
    submit_f.write(f"#SBATCH -n {ncpu}\n")
    submit_f.write("#SBATCH -N 1\n")
    submit_f.write("\n")
    submit_f.write(f"eval `cat {out_file} | head -${{SLURM_ARRAY_TASK_ID}} | tail -1`\n")
    submit_f.write("exit 0\n")

print("============== cmd example to submit prepare fragments jobs: ======================")
print(f"sbatch -a 1-{len(cmds_list)} {submit_script}")
print("===================================================================================")
