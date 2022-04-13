#!/usr/bin/env python
import os
import sys
sys.path.append("/home/huabai/scripts")

import gzip
import requests
import numpy as np
import pyrosetta
import sys
import re
from itertools import combinations, product

from utils.pose_utils import get_rotate_translate, transform_query_coordinates, get_xyz, set_xyz
from utils.catpi_utils import pipi_distance, pipi_angle
from utils.os_utils import create_dir


def check_and_remove_variant(input_pose, in_idx):
    upper = input_pose.residue(in_idx).has_variant_type(pyrosetta.rosetta.core.chemical.VariantType.UPPER_TERMINUS_VARIANT)
    if upper:
        pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( input_pose, pyrosetta.rosetta.core.chemical.UPPER_TERMINUS_VARIANT, in_idx)
    lower = input_pose.residue(in_idx).has_variant_type(pyrosetta.rosetta.core.chemical.VariantType.LOWER_TERMINUS_VARIANT)
    if lower:
        pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( input_pose, pyrosetta.rosetta.core.chemical.LOWER_TERMINUS_VARIANT, in_idx)
    return upper or lower


# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def pose_from_pdb(pdb_id, model_idx = 0, chain_id = None):
    url = "http://www.rcsb.org/pdb/files/{}.pdb".format(pdb_id.upper())
    r = requests.get(url, allow_redirects=True)
    model_pattern = re.compile(r"\nMODEL(.+?)\nENDMDL", re.DOTALL)
    models = model_pattern.findall(r.content.decode("utf-8"))

    if len(models) > 0:
        pdb_lines = models[model_idx].split("\n")
    else:
        pdb_lines = r.content.decode("utf-8").split("\n")

    if not chain_id:
        chosen_lines = pdb_lines
    else:
        ## 1 - 6 Record name "ATOM "
        ## 22 Character chainID Chain identifier
        chosen_lines = [x for x in pdb_lines if x[:6] == "ATOM  " and x[21:22] == chain_id]

    pdb_string = "\n".join(chosen_lines)
    loaded_pose = pyrosetta.Pose()

    if pdb_string:
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(loaded_pose, pdb_string)
    return loaded_pose

def pose_from_culled_lib(pdb_id, model_idx = 0, chain_id = None):
    culled_pdb_lib = "/projects/hashtables/prepare_hashtables/culled_pdbs"
    culled_pdb = f"{culled_pdb_lib}/{pdb_id.lower()}.pdb"

    with open(culled_pdb, "r") as f:
        raw_string = f.read()

    model_pattern = re.compile(r"\nMODEL(.+?)\nENDMDL", re.DOTALL)
    models = model_pattern.findall(raw_string)

    if len(models) > 0:
        pdb_lines = models[model_idx].split("\n")
    else:
        pdb_lines = raw_string.split("\n")

    if not chain_id:
        chosen_lines = pdb_lines
    else:
        ## 1 - 6 Record name "ATOM "
        ## 22 Character chainID Chain identifier
        chosen_lines = [x for x in pdb_lines if x[:6] == "ATOM  " and x[21:22] == chain_id]

    pdb_string = "\n".join(chosen_lines)
    loaded_pose = pyrosetta.Pose()

    if pdb_string:
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(loaded_pose, pdb_string)
    return loaded_pose

def mutate_residues_without_neighbors_consideration(inpose=None, mutant_positions=[], mutant_aas=[] ):
    test_pose = inpose.clone()

    for mutant_position, mutant_aa in zip(mutant_positions, mutant_aas):
        new_aa_name=pyrosetta.rosetta.core.chemical.name_from_aa(pyrosetta.rosetta.core.chemical.aa_from_oneletter_code( mutant_aa ))
        restype_set=test_pose.residue_type_set_for_pose() #test_pose.residue( mutant_position ) #.residue_type_set()
        new_res = pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue(rsd_type=restype_set.name_map(new_aa_name),
                                                                         current_rsd=test_pose.residue(mutant_position),
                                                                         conformation=test_pose.conformation(),
                                                                         preserve_c_beta=False)
        #Avoid disulfides for now
        if inpose.residue(mutant_position).has_variant_type(pyrosetta.rosetta.core.chemical.VariantType.DISULFIDE):
            continue
        test_pose.replace_residue( seqpos=mutant_position,
                                  new_rsd_in=new_res,
                                  orient_backbone=False );

        #N- or C-VARIANTS REMOVAL OR ADD (needed for proper cartesian minimization later)
        if inpose.residue(mutant_position).has_variant_type(pyrosetta.rosetta.core.chemical.VariantType.LOWER_TERMINUS_VARIANT):
            pyrosetta.rosetta.core.pose.add_variant_type_to_pose_residue(test_pose, pyrosetta.rosetta.core.chemical.VariantType.LOWER_TERMINUS_VARIANT,mutant_position)
        if inpose.residue(mutant_position).has_variant_type(pyrosetta.rosetta.core.chemical.VariantType.UPPER_TERMINUS_VARIANT):
            pyrosetta.rosetta.core.pose.add_variant_type_to_pose_residue(test_pose, pyrosetta.rosetta.core.chemical.VariantType.UPPER_TERMINUS_VARIANT,mutant_position)

    return test_pose

def res_list_to_pose(this_pose, res_list):
    res_list = sorted(res_list)

    fragments = []
    segment = []
    for x, y in zip(res_list[:-1], res_list[1:]):
        segment.append(x)
        if y-x > 2: ## separated by at least 2 residues
            fragments.append(segment)
            segment = []
    segment.append(res_list[-1])
    fragments.append(segment)

    segment_pose_list = []
    for each_segment in fragments:
        each_sub_pose = pyrosetta.Pose(this_pose, each_segment[0]-1, each_segment[-1]+1)

        range_list = [x for x in range(each_segment[0]-1, each_segment[-1]+2)]

        mutant_positions = [range_list.index(x)+1 for x in range_list if not x in each_segment]
        mutant_aas = ["G"]*len(mutant_positions)
        each_sub_pose = mutate_residues_without_neighbors_consideration(each_sub_pose, mutant_positions, mutant_aas)

        segment_pose_list.append(each_sub_pose)

    output_pose = None
    if segment_pose_list:
        output_pose = segment_pose_list[0].clone()
        for each_sub_pose in segment_pose_list[1:]:
            output_pose.append_pose_by_jump(each_sub_pose.clone(), 1)

    return output_pose

def res_list_to_BB_pose(this_pose, res_list):
    res_list = sorted(res_list)

    fragments = []
    segment = []
    for x, y in zip(res_list[:-1], res_list[1:]):
        segment.append(x)
        if y-x > 2: ## separated by at least 2 residues
            fragments.append(segment)
            segment = []
    segment.append(res_list[-1])
    fragments.append(segment)

    segment_pose_list = []
    for each_segment in fragments:
        each_sub_pose = pyrosetta.Pose(this_pose, each_segment[0]-1, each_segment[-1]+1)

        range_list = [x for x in range(each_segment[0]-1, each_segment[-1]+2)]

        new_core_idx = [range_list.index(x) + 1 for x in each_segment]

        # mutant_positions = [range_list.index(x)+1 for x in range_list if not x in each_segment]
        mutant_positions = [range_list.index(x)+1 for x in range_list]
        mutant_aas = ["G"]*len(mutant_positions)
        each_sub_pose = mutate_residues_without_neighbors_consideration(each_sub_pose, mutant_positions, mutant_aas)

        for each_core in new_core_idx:
            each_sub_pose.pdb_info().add_reslabel(each_core, "BB")

        segment_pose_list.append(each_sub_pose)

    output_pose = None
    if segment_pose_list:
        output_pose = segment_pose_list[0].clone()
        for each_sub_pose in segment_pose_list[1:]:
            output_pose.append_pose_by_jump(each_sub_pose.clone(), 1)

    return output_pose

def align_pose(input_pose):

    align_dict={}
    align_dict['W']=["CD2","CZ2","CZ3"]
    align_dict['Y']=["CG","CE1","CE2"]
    align_dict['F']=["CG","CE1","CE2"]
    align_dict['H']=["CG","ND1","NE2"]
    align_dict['R']=["NE","NH1","NH2"]
    align_dict['K']=["CD","NZ","CG"]
    align_dict['D']=["CG","OD1","OD2"]
    align_dict['N']=["CG","OD1","ND2"]
    align_dict['E']=["CD","OE1","OE2"]
    align_dict['Q']=["CD","OE1","NE2"]

    key_idx = 0
    for i in range(1, input_pose.size()+1):
        if input_pose.pdb_info().res_haslabel(i, "SC"):
            key_idx = i

    ## extract the xyz of the three atoms
    coordinates_mobile = get_xyz(input_pose, [key_idx], align_dict[input_pose.residue(key_idx).name1()])

    COM = coordinates_mobile[0]

    ## new xyz for coordinates_mobile[0]
    x1 = 0
    y1 = 0
    z1 = 0

    angle23 = angle_between(coordinates_mobile[1] - COM, coordinates_mobile[2] - COM)
    angle23 = angle23 / 2

    ## new xyz for coordinates_mobile[1]
    x2 = - np.linalg.norm(coordinates_mobile[1] - COM) * np.cos(angle23)
    y2 = np.linalg.norm(coordinates_mobile[1] - COM) * np.sin(angle23)
    z2 = 0

    ## new xyz for coordinates_mobile[2]
    x3 = - np.linalg.norm(coordinates_mobile[1] - COM) * np.cos(angle23)
    y3 = - np.linalg.norm(coordinates_mobile[1] - COM) * np.sin(angle23)
    z3 = 0

    ## the new xyz for CD or CG is [0,0,0]
    coordinates_ref = np.array([[x1, y1, z1],[x2, y2, z2],[x3, y3, z3]])
    assert(coordinates_mobile.shape == coordinates_ref.shape)

    rot_M, trans_V = get_rotate_translate(np.matrix(coordinates_mobile), np.matrix(coordinates_ref))

    old_mobile_pose_coordinates = get_xyz(input_pose)
    coordinates_output = transform_query_coordinates(np.matrix(old_mobile_pose_coordinates), rot_M, trans_V)
    set_xyz(input_pose, coordinates_output)

def generate_hbond_graph(input_pose):
    pose_len = input_pose.size()
    ## create and fill the hbond set
    hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    ## arguments: pose: pyrosetta.rosetta.core.pose.Pose, calculate_derivative: bool, backbone_only: bool
    ##  hbset.setup_for_residue_pair_energies(input_pose, False, False) ## use 3rd argument False to include all Hbonds
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(input_pose, calculate_derivative=False, hbond_set=hbset)

    ## create hbond graph
    dt = np.dtype([("DON_KRNQ", "i1"), ("ACC_DENQ", "i1"), ("DON_WYH", "i1")])
    hbond_graph=np.zeros((pose_len, pose_len), dtype=dt)

    ## go through all residues
    for res_i in range(1, pose_len+1):
        for hbond in hbset.residue_hbonds(res_i):

            don_idx = hbond.don_res()
            don_atom = hbond.don_hatm()
            don_res = input_pose.residue(don_idx)
            don_atom_name = don_res.atom_name(don_atom)
            don_name3 = don_res.name3()

            acc_idx = hbond.acc_res()
            acc_atom = hbond.acc_atm()
            acc_res = input_pose.residue(acc_idx)
            acc_atom_name = acc_res.atom_name(acc_atom)
            acc_name3 = acc_res.name3()

            if don_idx == res_i and hbond.don_hatm_is_backbone(): continue
            if acc_idx == res_i and hbond.acc_atm_is_backbone(): continue

            if don_name3 in ["LYS", "ARG", "GLN", "ASN"] and (not hbond.don_hatm_is_backbone()) and hbond.acc_atm_is_backbone():
                # print("Res{} {} {} --> Res{} {} {}".format(don_idx, don_name3, don_atom_name, acc_idx, acc_name3, acc_atom_name))
                hbond_graph["DON_KRNQ"][don_idx-1, acc_idx-1] += 1

            if acc_name3 in ["GLU", "ASP", "ASN", "GLN"] and (not hbond.acc_atm_is_backbone()) and hbond.don_hatm_is_backbone():
                # print("Res{} {} {} --> Res{} {} {}".format(don_idx, don_name3, don_atom_name, acc_idx, acc_name3, acc_atom_name))
                hbond_graph["ACC_DENQ"][don_idx-1, acc_idx-1] += 1

            if don_name3 in ["TRP", "TYR", "HIS"] and (not hbond.don_hatm_is_backbone()) and hbond.acc_atm_is_backbone():
                # print("Res{} {} {} --> Res{} {} {}".format(don_idx, don_name3, don_atom_name, acc_idx, acc_name3, acc_atom_name))
                hbond_graph["DON_WYH"][don_idx-1, acc_idx-1] += 1

    return hbond_graph, hbset



def NQ_BB_worker(input_info, input_pose, out_dir):
    this_pdb = input_info[0]
    this_chain = input_info[1]

    SCBB_pairs = []

    hbond_graph, hbond_set = generate_hbond_graph(input_pose)
    # print(hbond_graph[np.nonzero(hbond_graph)])

    ## search residues with at least two hbonds
    don_sum = np.sum(hbond_graph["DON_KRNQ"], axis=1)
    acc_sum = np.sum(hbond_graph["ACC_DENQ"], axis=0)

    ## the residue act both receptor and donor
    power_hybrid = [x+1 for x in np.where( np.logical_and(don_sum == 1, acc_sum == 1 ))[0] ]
    for sc_idx in power_hybrid:
        bb_acc_array = hbond_graph["DON_KRNQ"][sc_idx-1,:]
        bb_acc_positions = [ x+1 for x in np.where(bb_acc_array >= 1)[0] ]
        bb_don_array = hbond_graph["ACC_DENQ"][:,sc_idx-1]
        bb_don_positions = [ x+1 for x in np.where(bb_don_array >= 1)[0] ]

        if len(bb_acc_positions) > 1 or len(bb_don_positions) > 1: continue
        if not bb_acc_positions[0] == bb_don_positions[0]: continue

        sc_name1 = input_pose.residue(sc_idx).name1()
        if not sc_name1 in ["N", "Q"]: continue

        bb_idx = bb_acc_positions[0]
        if abs(sc_idx - bb_idx) < 3: continue

        if bb_idx < 3 or bb_idx > input_pose.size()-2: continue

        print(f"Power hybrid residue: {sc_idx} {input_pose.residue(sc_idx).name3()}")
        print( f"Acc positions: {bb_acc_positions}" )
        print( f"Acc hbond count: {bb_acc_array[bb_acc_array >= 1]}" )
        print( f"Don positions: {bb_don_positions}" )
        print( f"Don hbond count: {bb_don_array[bb_don_array >= 1]}" )

        ## check the atom names on sc_idx
        sc_atom_names = []
        for hbond in hbond_set.residue_hbonds(sc_idx):
            don_idx = hbond.don_res()
            don_atom = hbond.don_hatm()
            don_res = input_pose.residue(don_idx)
            don_atom_name = don_res.atom_name(don_atom)
            don_name3 = don_res.name3()

            acc_idx = hbond.acc_res()
            acc_atom = hbond.acc_atm()
            acc_res = input_pose.residue(acc_idx)
            acc_atom_name = acc_res.atom_name(acc_atom)
            acc_name3 = acc_res.name3()

            if don_idx == sc_idx and don_name3 in ["GLN", "ASN"] and (not hbond.don_hatm_is_backbone()) and hbond.acc_atm_is_backbone():
                sc_atom_names.append(don_atom_name.strip())
            if acc_idx == sc_idx and acc_name3 in ["ASN", "GLN"] and (not hbond.acc_atm_is_backbone()) and hbond.don_hatm_is_backbone():
                sc_atom_names.append(acc_atom_name.strip())

        sc_atom_names  = list(set(sc_atom_names))
        sc_atom_names.sort()
        sc_atom_n = len(sc_atom_names)
        if sc_atom_n != 2: continue
        sc_atom_names  = "_".join(sc_atom_names)

        pair_id = f"{sc_name1}_{sc_atom_names}"
        sub_dir = f"{out_dir}/{pair_id}"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        sc_pose = pyrosetta.Pose(this_pose, sc_idx-1, sc_idx+1)
        mutant_positions = [1, 3]
        mutant_aas = ["G"]*len(mutant_positions)
        sc_pose = mutate_residues_without_neighbors_consideration(sc_pose, mutant_positions, mutant_aas)
        sc_pose.pdb_info().add_reslabel(2, "SC")

        bb_pose = pyrosetta.Pose(this_pose, bb_idx-1, bb_idx+1)
        mutant_positions = [1, 2, 3]
        mutant_aas = ["G"]*len(mutant_positions)
        bb_pose = mutate_residues_without_neighbors_consideration(bb_pose, mutant_positions, mutant_aas)
        bb_pose.pdb_info().add_reslabel(2, "BB")

        output_pose = sc_pose.clone()
        output_pose.append_pose_by_jump(bb_pose.clone(), 1)

        sc_chain_pdb_idx = this_pose.pdb_info().pose2pdb(sc_idx).split()[0]
        bb_chain_pdb_idx = this_pose.pdb_info().pose2pdb(bb_idx).split()[0]
        output_id = f"{sub_dir}/NQBB_{pair_id}_{this_pdb}_{this_chain}_{sc_chain_pdb_idx}_{bb_chain_pdb_idx}.pdb"

        align_pose(output_pose)

        for ii in range(1, output_pose.size()+1):
            check_and_remove_variant(output_pose, ii)

        output_pose.dump_pdb(output_id)

if __name__ == "__main__":

    pyrosetta.init("-beta -mute all")

    input_gz = sys.argv[1]
    if input_gz.endswith(".gz"):
        with gzip.open(input_gz, "rb") as input_f:
            pdb_list = [x.decode("utf-8").strip().split()[0] for x in input_f.readlines()[1:]]
    else:
        with open(input_gz, "r") as f:
            pdb_list = [x.strip().split()[0] for x in f.readlines()[1:]]

    start_line = int(sys.argv[2])
    end_line = int(sys.argv[3])
    out_dir = sys.argv[4]
    create_dir(out_dir)

    for i in range(start_line, end_line):
        this_pdb = pdb_list[i][:4].lower()
        this_chain = pdb_list[i][4]

        this_pdb = pdb_list[i][:4]
        this_chain = pdb_list[i][4]
        print(f"======================== {this_pdb} chain {this_chain}:=====================")
        try:
            this_pose = pose_from_culled_lib(this_pdb, chain_id = this_chain)
        except Exception:
            print("cannot get PDB!!")
            continue

        this_pose.update_residue_neighbors()

        ## =========== get the hbond information =================
        ## prepare scorefxn
        scorefxn = pyrosetta.get_score_function()
        ## score the pose

        try:
            scorefxn(this_pose)
        except Exception:
            continue

        this_info = (this_pdb, this_chain)
        try:
            NQ_BB_worker(this_info, this_pose, out_dir)
        except Exception:
            continue
