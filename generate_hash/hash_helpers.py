#!/usr/bin/env python
import requests
import re
from collections import defaultdict
from itertools import product
import os
import pyrosetta
import numpy as np
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3
from enum import Enum
import rif.hash
from rif.geom.ray_hash import RayRay10dHash
from rif.geom import Ray

def get_SCRR_enum():
    SC_choices = ["D", "N", "E", "Q", "K", "R", "H", "S", "T", "W", "Y"]
    rayray_choices = [ ("NH", "CO"), ("CO", "NH"), ("NH", "NH"), ("CO", "CO"), ("NH", ""), ("CO", "") ]
    potential_SCRR_list = [ f"{each_SCRR[0]}{each_SCRR[1][0]}{each_SCRR[1][1]}" for each_SCRR in product(SC_choices, rayray_choices) ]
    assert(len(potential_SCRR_list) <= 255)
    ## SCRR: sidechain-rayrary
    SCRR_enum = Enum("SCRR", potential_SCRR_list)
    return SCRR_enum

def check_hbond_types(hbset, aa_trio, res_idx):
    hbonds = hbset.residue_hbonds( res_idx )
    hbond_type = []
    for hbond in hbonds:
        don_idx = hbond.don_res()
        don_atom = hbond.don_hatm() ## mean the H, not heavyatom
        don_res = aa_trio.residue(don_idx)
        don_atom_name = don_res.atom_name(don_atom)
        don_name3 = don_res.name3()

        acc_idx = hbond.acc_res()
        acc_atom = hbond.acc_atm()
        acc_res = aa_trio.residue(acc_idx)
        acc_atom_name = acc_res.atom_name(acc_atom)
        acc_name3 = acc_res.name3()

        ## trick: use strip to remove the white space around atom names
        if don_idx == res_idx and hbond.don_hatm_is_backbone() \
                              and don_atom_name.strip() == "H" \
                              and not hbond.acc_atm_is_backbone():
            hbond_type.append("NH")
        if acc_idx == res_idx and hbond.acc_atm_is_backbone() \
                              and acc_atom_name.strip() == "O" \
                              and not hbond.don_hatm_is_backbone():
            hbond_type.append("CO")

    return list(set(hbond_type))

def check_ray_type(input_pose, SC_idx, BB_idx_list):
    scorefxn = pyrosetta.create_score_function("ref2015")
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)
    try:
        scorefxn(input_pose)
    except Exception:
        return []

    hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(input_pose, calculate_derivative=False, hbond_set=hbset)

    if len(BB_idx_list) == 1:
        # print( check_hbond_types(hbset, input_pose, BB_idx_list[0]) )
        # print( input_pose.residue(SC_idx).name1() )
        if not len(check_hbond_types(hbset, input_pose, BB_idx_list[0])) == 2:
            # print("too few hbond?")
            return []

        Aray_str = "NH"
        Bray_str = "CO"

    else:
        Nterm_type = check_hbond_types(hbset, input_pose, BB_idx_list[0] )
        Cterm_type = check_hbond_types(hbset, input_pose, BB_idx_list[1] )
        try:
            assert (len(Nterm_type) == 1 and len(Cterm_type) == 1)
        except:
            # print("too much hbond?")
            return []

        Aray_str = Nterm_type[0]
        Bray_str = Cterm_type[0]

    return [Aray_str, Bray_str]


def V2_check_ray_type(input_pose, SC_idx, BB_idx_list):
    scorefxn = pyrosetta.create_score_function("ref2015")
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)
    try:
        scorefxn(input_pose)
    except Exception:
        return None

    hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(input_pose, calculate_derivative=False, hbond_set=hbset)

    if len(BB_idx_list) == 1:
        # print( check_hbond_types(hbset, input_pose, BB_idx_list[0]) )
        # print( input_pose.residue(SC_idx).name1() )
        if not len(check_hbond_types(hbset, input_pose, BB_idx_list[0])) == 2:
            # print("too few hbond?")
            return None

        Aray_str = "NH"
        Bray_str = "CO"
        Aray_idx = BB_idx_list[0]
        Bray_idx = BB_idx_list[0]

    else:
        Nterm_type = check_hbond_types(hbset, input_pose, BB_idx_list[0] )
        Cterm_type = check_hbond_types(hbset, input_pose, BB_idx_list[1] )
        try:
            assert (len(Nterm_type) == 1 and len(Cterm_type) == 1)
        except:
            # print("too much hbond?")
            return None

        Aray_str = Nterm_type[0]
        Bray_str = Cterm_type[0]

        Aray_idx = BB_idx_list[0]
        Bray_idx = BB_idx_list[1]


    if Aray_str == "CO" and Bray_str == "NH":
        Aray_str = "NH"
        Bray_str = "CO"

        Aray_idx = BB_idx_list[1]
        Bray_idx = BB_idx_list[0]

    return (Aray_idx, Aray_str), (Bray_idx, Bray_str)

def pose_from_pdb(pdb_id, model_idx = 0, chains = "A", culled_pdb_lib = "/home/huabai/DBs/culled_pdbs"):
    culled_pdb = f"{culled_pdb_lib}/{pdb_id.lower()}.pdb"

    if os.path.isfile(culled_pdb):
        with open(culled_pdb, "r") as f:
            raw_string = f.read()
    else:
        url = "http://www.rcsb.org/pdb/files/{}.pdb".format(pdb_id.upper())
        r = requests.get(url, allow_redirects=True)
        raw_string = r.content.decode("utf-8")

    model_pattern = re.compile(r"\nMODEL(.+?)\nENDMDL", re.DOTALL)
    models = model_pattern.findall(raw_string)

    if len(models) > 0:
        pdb_lines = models[model_idx].split("\n")
    else:
        pdb_lines = raw_string.split("\n")

    ## create a dictionary of the original information, and remove the redundant chain ID.
    pdb_lines_dict = defaultdict(lambda: defaultdict(list))
    output_lines = []
    for each_line in pdb_lines:
        # if each_line.startswith("ATOM") or each_line.startswith("HETATM"):
        if each_line.startswith("ATOM"):
            ## remove alternative conformation, just for simplicity
            ## can handel multiple conformation in the future
            altLoc = each_line[16].strip()
            if not altLoc == "" and not altLoc == "A":
                continue

            chain_id = each_line[21]
            res_id   = each_line[17:20].strip()
            res_idx = int(each_line[22:26])
            new_line = each_line[:66] + " "*(76-66) + each_line[76:] ##
            pdb_lines_dict[chain_id][(res_idx,res_id)].append(new_line)

            if chain_id in chains:
                output_lines.append(new_line)

    pdb_string = "\n".join(output_lines)
    loaded_pose = pyrosetta.Pose()
    if pdb_string:
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(loaded_pose, pdb_string)

    return loaded_pose, pdb_lines_dict

def create_hbond_graph(input_pose):
    sfx = pyrosetta.create_score_function("ref2015")
    sfx(input_pose)
    hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(input_pose, calculate_derivative=False, hbond_set=hbset)

    pose_len = input_pose.size()
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

def DNEQ_BB_worker(input_info, input_pose, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    this_pdb = input_info[0]
    this_chain = input_info[1]

    DENQKR_BB_pairs = []

    hbond_graph, hbond_set = create_hbond_graph(input_pose)
    # print(hbond_graph[np.nonzero(hbond_graph)])

    ## search residues with at least two hbonds
    don_sum = np.sum(hbond_graph["DON_KRNQ"], axis=1)
    acc_sum = np.sum(hbond_graph["ACC_DENQ"], axis=0)

    power_don = [ x+1 for x in np.where( np.logical_and(don_sum ==2 , acc_sum==0) )[0] ]
    for res_idx in power_don:
        # print(f"Power doner residue: {res_idx} {input_pose.residue(res_idx).name3()}")
        bb_acc_array = hbond_graph["DON_KRNQ"][res_idx-1,:]
        bb_acc_positions = [ x+1 for x in np.where(bb_acc_array >= 1)[0] ]
        DENQKR_BB_pairs.append( ("KR-->BB", res_idx, [], bb_acc_positions) )

        # print( f"Acc positions: {bb_acc_positions}" )
        # print( f"Acc hbond count: {bb_acc_array[bb_acc_array >= 1]}" )

    power_acc = [ x+1 for x in np.where( np.logical_and(acc_sum ==2 , don_sum==0) )[0] ]
    for res_idx in power_acc:
        # print(f"Power acceptor residue: {res_idx} {input_pose.residue(res_idx).name3()}")
        bb_don_array = hbond_graph["ACC_DENQ"][:,res_idx-1]
        bb_don_positions = [ x+1 for x in np.where(bb_don_array >= 1)[0] ]
        DENQKR_BB_pairs.append( ("BB-->DE", res_idx, bb_don_positions, []) )

        # print( f"Don positions: {bb_don_positions}" )
        # print( f"Don hbond count: {bb_don_array[bb_don_array >= 1]}" )

    ## the residue act both receptor and donor
    power_hybrid = [x+1 for x in np.where( np.logical_and(don_sum == 1, acc_sum == 1 ))[0] ]
    for res_idx in power_hybrid:
        # print(f"Power hybrid residue: {res_idx} {input_pose.residue(res_idx).name3()}")
        bb_acc_array = hbond_graph["DON_KRNQ"][res_idx-1,:]
        bb_acc_positions = [ x+1 for x in np.where(bb_acc_array >= 1)[0] ]
        bb_don_array = hbond_graph["ACC_DENQ"][:,res_idx-1]
        bb_don_positions = [ x+1 for x in np.where(bb_don_array >= 1)[0] ]

        DENQKR_BB_pairs.append( ("BB-->NQ-->BB", res_idx, bb_don_positions, bb_acc_positions) )
        # print( f"Acc positions: {bb_acc_positions}" )
        # print( f"Acc hbond count: {bb_acc_array[bb_acc_array >= 1]}" )
        # print( f"Don positions: {bb_don_positions}" )
        # print( f"Don hbond count: {bb_don_array[bb_don_array >= 1]}" )

    for pair_info in DENQKR_BB_pairs:
        don_acc_type, sc_idx, bb_don_res_list, bb_acc_res_list = pair_info
        bb_res_list = bb_don_res_list + bb_acc_res_list
        bb_res_list = list(set(bb_res_list))
        bb_res_list.sort()
        try:
            ## sc res not close to the start or the end of pose
            assert sc_idx > 1 and sc_idx < input_pose.size()
            for j in bb_res_list:
                ## not close to the start or the end of pose
                assert j > 1 and j < input_pose.size()
                ## make sure the sc res and bb res are not too close
                assert abs(sc_idx - j) > 2 ## may change to >= 2
        except AssertionError:
            print(f"too close to terminal or each other: {sc_idx} {bb_res_list}; skip")
            continue

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

            ## case: the sc_idx's side chain is the acceptor
            if don_acc_type == "BB-->DE" and don_idx == sc_idx: continue
            if don_acc_type == "BB-->DE" and hbond.acc_atm_is_backbone(): continue
            ## case: the sc_idx's side chain is the doner
            if don_acc_type == "KR-->BB" and acc_idx == sc_idx: continue
            if don_acc_type == "KR-->BB" and hbond.don_hatm_is_backbone(): continue

            if don_idx == sc_idx and don_name3 in ["LYS", "ARG", "GLN", "ASN"] and (not hbond.don_hatm_is_backbone()) and hbond.acc_atm_is_backbone():
                # print("Res{} {} {} --> Res{} {} {}".format(don_idx, don_name3, don_atom_name, acc_idx, acc_name3, acc_atom_name))
                sc_atom_names.append(don_atom_name.strip())
            if acc_idx == sc_idx and acc_name3 in ["GLU", "ASP", "ASN", "GLN"] and (not hbond.acc_atm_is_backbone()) and hbond.don_hatm_is_backbone():
                # print("Res{} {} {} --> Res{} {} {}".format(don_idx, don_name3, don_atom_name, acc_idx, acc_name3, acc_atom_name))
                sc_atom_names.append(acc_atom_name.strip())

        sc_atom_names  = list(set(sc_atom_names))
        sc_atom_names.sort()
        sc_atom_n = len(sc_atom_names)
        if sc_atom_n != 2: continue

        sc_atom_names  = "_".join(sc_atom_names)

        bb_res_names = "".join( [input_pose.residue(x).name1() for x in bb_res_list] )
        bb_don_res_names = "".join( [input_pose.residue(x).name1() for x in bb_don_res_list] )
        bb_acc_res_names = "".join( [input_pose.residue(x).name1() for x in bb_acc_res_list] )

        # print(sc_idx)
        sc_chain_pdb_idx = int( input_pose.pdb_info().pose2pdb(sc_idx).split()[0] )

        # print(bb_res_list)
        triad_pose = generate_SCRR_pose(input_pose, sc_idx, bb_res_list)
        # print(triad_pose)

        print(triad_pose)
        if len(bb_res_list) == 2:
            ray_types = check_ray_type(triad_pose, 1, [2,3])
        elif len(bb_res_list) == 1:
            ray_types = check_ray_type(triad_pose, 1, [2])
        else:
            continue

        if not ray_types: continue
        # print(ray_types)
        ray_id = f"{input_pose.residue(sc_idx).name1()}{ray_types[0]}{ray_types[1]}"

        res_diff = abs(bb_res_list[0] - bb_res_list[-1])

        # assert len(bb_res_list) in [1,2]

        sub_dir = f"{out_dir}/{ray_id}"

        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        output_id = f"{sub_dir}/{ray_id}_N{res_diff}_{this_pdb}_chain{this_chain}_pdbidx_{sc_chain_pdb_idx}.pdb"

        align_pose(triad_pose)
        triad_pose.dump_pdb(output_id)

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

def check_connection(this_pose, ref_idx, query_idx):
    check_lower = this_pose.residue(ref_idx).connected_residue_at_lower() == query_idx
    check_upper = this_pose.residue(ref_idx).connected_residue_at_upper() == query_idx
    return check_lower or check_upper

def check_and_remove_variant(input_pose, in_idx):
    upper = input_pose.residue(in_idx).has_variant_type(pyrosetta.rosetta.core.chemical.VariantType.UPPER_TERMINUS_VARIANT)
    if upper:
        pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( input_pose, pyrosetta.rosetta.core.chemical.UPPER_TERMINUS_VARIANT, in_idx)
    lower = input_pose.residue(in_idx).has_variant_type(pyrosetta.rosetta.core.chemical.VariantType.LOWER_TERMINUS_VARIANT)
    if lower:
        pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( input_pose, pyrosetta.rosetta.core.chemical.LOWER_TERMINUS_VARIANT, in_idx)
    return upper or lower

def generate_SCRR_pose(this_pose, sc_idx, bb_res_list):
    out_pose = pyrosetta.Pose()
    out_pose.append_residue_by_jump(this_pose.residue(sc_idx).clone(), 1)



    if len(bb_res_list) == 1:
        bb_idx_A = bb_res_list[0]
        assert not check_connection(this_pose, sc_idx, bb_idx_A)
        out_pose.append_residue_by_jump(this_pose.residue(bb_idx_A).clone(), 1)

        out_info = pyrosetta.rosetta.core.pose.PDBInfo(out_pose)
        out_pose.pdb_info(out_info)
        out_pose.pdb_info().add_reslabel(1, "SC")
        out_pose.pdb_info().add_reslabel(2, "BB")

    else:
        assert len(bb_res_list) == 2
        bb_idx_A = bb_res_list[0]
        bb_idx_B = bb_res_list[1]
        assert not check_connection(this_pose, sc_idx, bb_idx_A)
        assert not check_connection(this_pose, sc_idx, bb_idx_B)

        if not check_connection(this_pose, bb_idx_A, bb_idx_B):
            out_pose.append_residue_by_jump(this_pose.residue(bb_idx_A).clone(), 1)
            out_pose.append_residue_by_jump(this_pose.residue(bb_idx_B).clone(), 1)

        else:
            check_and_remove_variant(this_pose, bb_idx_A)
            check_and_remove_variant(this_pose, bb_idx_B)
            out_pose.append_residue_by_jump(this_pose.residue(bb_idx_A).clone(), 1)
            out_pose.append_residue_by_bond(this_pose.residue(bb_idx_B).clone())

        out_info = pyrosetta.rosetta.core.pose.PDBInfo(out_pose)
        out_pose.pdb_info(out_info)
        out_pose.pdb_info().add_reslabel(1, "SC")
        out_pose.pdb_info().add_reslabel(2, "BB")
        out_pose.pdb_info().add_reslabel(3, "BB")

    return out_pose

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

    # print(key_idx)

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

def get_xyz(input_pose, residue_range = "all", atom_names = "all"):
    if residue_range == "all":
        residue_range = range(1, input_pose.size()+1)
    output_xyz_list = []
    for ri in residue_range:
        if atom_names == "all":
            for ai in range(1, input_pose.residue(ri).natoms()+1):
                xyz = input_pose.residue(ri).xyz(ai)
                output_xyz_list.append([xyz[0], xyz[1], xyz[2]])
        elif atom_names == "heavy":
            for ai in range(1, input_pose.residue(ri).nheavyatoms()+1):
                xyz = input_pose.residue(ri).xyz(ai)
                output_xyz_list.append([xyz[0], xyz[1], xyz[2]])

        elif atom_names == "sc":
            heavy_idx_set = set([x for x in range(1, input_pose.residue(ri).nheavyatoms()+1)])
            bb_idx_set = set(input_pose.residue(ri).all_bb_atoms())
            res_sc_heavy = list(heavy_idx_set - bb_idx_set)
            for ai in res_sc_heavy:
                xyz = input_pose.residue(ri).xyz(ai)
                output_xyz_list.append([xyz[0], xyz[1], xyz[2]])

        else:
            for an in atom_names:
                ## use atom_name in atom_names could be better
                ## atom_name = ref_pose.residue(ir).atom_name(ia).strip()
                xyz = input_pose.residue(ri).xyz(an)
                output_xyz_list.append([xyz[0], xyz[1], xyz[2]])
    return np.array(output_xyz_list)

def set_xyz(input_pose, new_coords):
    idx = 0
    for ir in range(1, input_pose.size() + 1):
        for ia in range(1, input_pose.residue(ir).natoms() + 1):
            vector = V3(new_coords[idx][0], new_coords[idx][1], new_coords[idx][2])
            input_pose.residue(ir).set_xyz(ia, vector)
            idx += 1
    return input_pose

## B = A * R + t
## A B example: [[point1.x, point1.y, point1.z],
##               [point2.x, point2.y, point2.z],
##               ......
##               [pointN.x, pointN.y, pointN.z]]
def get_rotate_translate(a,b):
    assert a.shape == b.shape, "check the dimensions of the input matrix"
    # Get the coordinates centroids
    mean_a = np.mean(a,axis=0)
    mean_b = np.mean(b,axis=0)
    # Normalized matrices
    norm_a = a - mean_a
    norm_b = b - mean_b
    # Generate cross dispersion matrix for a and b
    cross_dispersion = norm_a.T * norm_b

    # Do the SVD to get rotation matrix
    u,s,v = np.linalg.svd(cross_dispersion)
    # Rotation matrix
    R_m = (v.T * u.T).T
    # Due to reflection check determinant of matrix det A = -1
    if np.linalg.det(R_m) < 0:
        v[2] = -v[2]
        R_m = (v.T * u.T).T
    trans_V = mean_b - mean_a * R_m
    return R_m, trans_V

def transform_query_coordinates(a, rotation_matrix, translation_matrix):
    b = a * rotation_matrix + translation_matrix
    b = np.asarray(b)
    return b




def hbonds_between_aa_trio(aa_trio, SC_idx, BB_idx_list, scorefxn=None):
    if not scorefxn:
        scorefxn = pyrosetta.create_score_function("ref2015")
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)

    score_trio = scorefxn(aa_trio)
    hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    ## arguments: pose: pyrosetta.rosetta.core.pose.Pose, calculate_derivative: bool, backbone_only: bool
    hbond_set.setup_for_residue_pair_energies(aa_trio, False, False)
    n_hbonds = hbond_set.nhbonds()

    hbond_pattern_value = 0
    SC_hbond_count = 0
    SC_hbond_atoms = []
    hbond_type = []
    for hbond in hbond_set.residue_hbonds(SC_idx):
        acc_idx = hbond.acc_res()
        acc_atom = hbond.acc_atm()
        acc_res = aa_trio.residue(acc_idx)
        acc_atom_name = acc_res.atom_name(acc_atom).strip()
        acc_name3 = acc_res.name3()

        don_idx = hbond.don_res()
        don_atom = hbond.don_hatm()
        don_res = aa_trio.residue(don_idx)
        don_atom_name = don_res.atom_name(don_atom)
        don_name3 = don_res.name3()
        don_heavyatom_idx = aa_trio.residue(don_idx).get_adjacent_heavy_atoms(don_atom)[1]
        don_heavyatom_name = aa_trio.residue(don_idx).atom_name(don_heavyatom_idx).strip()

        each_eval_tuple = hbond.eval_tuple()

        if acc_idx == SC_idx and (not hbond.acc_atm_is_backbone()) and\
           don_idx in BB_idx_list and hbond.don_hatm_is_backbone():
            # print("Res{} {} {} --> Res{} {} {}".format(don_idx, don_name3, don_heavyatom_name, acc_idx, acc_name3, acc_atom_name))
            SC_hbond_count += 1
            SC_hbond_atoms.append(acc_atom_name)
            hbond_pattern_value += ( int( each_eval_tuple.don_type() ) +
                                     int( each_eval_tuple.acc_type() ) +
                                     int( each_eval_tuple.sequence_sep() ) +
                                     int( each_eval_tuple.eval_type() ) )
            hbond_type.append("NH")

        if don_idx == SC_idx and (not hbond.don_hatm_is_backbone()) and\
           acc_idx in BB_idx_list and hbond.acc_atm_is_backbone():
            # print("Res{} {} {} --> Res{} {} {}".format(don_idx, don_name3, don_heavyatom_name, acc_idx, acc_name3, acc_atom_name))
            SC_hbond_count += 1
            SC_hbond_atoms.append(don_heavyatom_name)
            hbond_pattern_value += ( int( each_eval_tuple.don_type() ) +
                                     int( each_eval_tuple.acc_type() ) +
                                     int( each_eval_tuple.sequence_sep() ) +
                                     int( each_eval_tuple.eval_type() ) )
            hbond_type.append("CO")

    return SC_hbond_count, set(SC_hbond_atoms), hbond_pattern_value


def get_rots_for_single_residue(ori_res, ori_phi=-66, ori_psi=-42, scorefxn=None, include_current=True):
    ## init scorefxn if not provided
    if not scorefxn:
        scorefxn = pyrosetta.create_score_function("ref2015")
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)

    ## create a pose with just the input residue
    pose_for_rots = pyrosetta.Pose()
    pose_for_rots.append_residue_by_jump(ori_res.clone(), 1)
    pose_for_rots.set_phi(1, ori_phi)
    pose_for_rots.set_phi(1, ori_psi)
    pose_for_rots.set_omega(1, 180.0)

    packTask=pyrosetta.rosetta.core.pack.task.TaskFactory.create_packer_task(pose_for_rots)
    packTask.set_bump_check(True)

    resTask=packTask.nonconst_residue_task(1)
    resTask.or_include_current(include_current)
    resTask.restrict_to_repacking()
    resTask.or_ex1( True )
    resTask.or_ex2( True )

    packer_graph=pyrosetta.rosetta.utility.graph.Graph(1)
    rsf=pyrosetta.rosetta.core.pack.rotamer_set.RotamerSetFactory()
    rotset_ = rsf.create_rotamer_set( pose_for_rots )
    rotset_.set_resid(1)

    ## the last parameter False means use_neighbor_context = False ## the name is confusing
    ## which will eventually effect the ResidueLevelTask_::extrachi_sample_level method
    rotset_.build_rotamers(pose_for_rots, scorefxn, packTask, packer_graph, False )

    # print(f"Found {rotset_.num_rotamers()} rotamers, with ex-levels ex1_ex2")
    rotamer_set=[]

    if include_current:
        rotamer_set.append(ori_res.clone())

    for irot in range(1,rotset_.num_rotamers()+1):
        this_rotamer = rotset_.rotamer(irot).clone()
        if not this_rotamer.is_similar_rotamer(ori_res.clone()):
            rotamer_set.append(this_rotamer)
    return rotamer_set



class aa_trio_hash_base:
    def __init__(self, resl, lever):
        print(f"Generating hashing function with parameters")
        self.resl=resl
        self.lever=lever
        self.hash_function = RayRay10dHash(resl, lever)

    def get_stub(self, res, stub_atoms = ["N", "CA", "C"]):
        from pyrosetta.rosetta.numeric import xyzVector_double_t as V3
        return pyrosetta.rosetta.core.kinematics.Stub( V3(res.xyz(stub_atoms[0])),
                                                       V3(res.xyz(stub_atoms[1])),
                                                       V3(res.xyz(stub_atoms[2])) )

    def get_hash_key(self, ref_res, ray_A_res, ray_B_res,
                           stub_atoms = ["N", "CA", "C"],
                           ray_A_atoms = ["N", "H"],
                           ray_B_atoms = ["C", "O"] ):

        ref_stub = self.get_stub(ref_res, stub_atoms)
        ray_A_start = ref_stub.global2local( ray_A_res.xyz( ray_A_atoms[0] ) )
        ray_A_end   = ref_stub.global2local( ray_A_res.xyz( ray_A_atoms[1] ) )
        ray_B_start = ref_stub.global2local( ray_B_res.xyz( ray_B_atoms[0] ) )
        ray_B_end   = ref_stub.global2local( ray_B_res.xyz( ray_B_atoms[1] ) )
        ray_A = Ray( orig=ray_A_start, dirn=( ray_A_end - ray_A_start ) )
        ray_B = Ray( orig=ray_B_start, dirn=( ray_B_end - ray_B_start ) )
        rayray_hash_key = self.hash_function.get_key( ray_A, ray_B )
        return rayray_hash_key


    def get_rots_GXG(self, GXG_pose, scorefxn=None, ex1=False, ex2=False):
        if not scorefxn:
            scorefxn = pyrosetta.create_score_function("ref2015")
            scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
            scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)

        SC_idx = 2
        ori_res = GXG_pose.residue(2).clone()

        ## create a pose with just the input residue
        pose_for_rots = GXG_pose.clone()
        packTask=pyrosetta.rosetta.core.pack.task.TaskFactory.create_packer_task(pose_for_rots)
        packTask.set_bump_check(True)

        resTask=packTask.nonconst_residue_task(SC_idx)
        resTask.or_include_current(True)
        resTask.restrict_to_repacking()
        resTask.or_ex1( ex1 )
        resTask.or_ex2( ex2 )

        packer_graph=pyrosetta.rosetta.utility.graph.Graph(GXG_pose.size())
        rsf=pyrosetta.rosetta.core.pack.rotamer_set.RotamerSetFactory()
        rotset_ = rsf.create_rotamer_set( pose_for_rots )
        rotset_.set_resid(SC_idx)
        ## the last parameter False means use_neighbor_context = False ## the name is confusing
        ## which will eventually effect the ResidueLevelTask_::extrachi_sample_level method
        rotset_.build_rotamers(pose_for_rots, scorefxn, packTask, packer_graph, False )
        print(f"Found {rotset_.num_rotamers()} rotamers, with ex-levels ex1 {ex1} ex2 {ex2}")

        rotamer_set=[]
        rotamer_set.append(ori_res.clone())

        for irot in range(1,rotset_.num_rotamers()+1):
            this_rotamer = rotset_.rotamer(irot).clone()
            if not this_rotamer.is_similar_rotamer(ori_res.clone()):
                rotamer_set.append(this_rotamer)
        return rotamer_set

    def generate_aligned_rotamers_single_residue(self, ori_res, ori_phi=-66, ori_psi=-42, side=0, ref_res=None, scorefxn=None):
        if not ref_res:
            ref_res = ori_res.clone()

        alignment_atoms = self.get_alignment_dict()[ori_res.name1()][side]

        alignment_vector = pyrosetta.rosetta.utility.vector1_std_pair_std_string_std_string_t()
        for atom in alignment_atoms:
            alignment_vector.append( (atom, atom) )

        ## generate raw rotamers, not aligned yet
        raw_rots=self.get_rots(ori_res, ori_phi, ori_psi, scorefxn)
        print(f"Number of inverse rotamers used in hashing for res {ori_res.name3()}: {len(raw_rots)}")

        aligned_rots=[]
        for rot in raw_rots:
            aligned_rot = rot.clone()
            # print("pre_align")
            aligned_rot.orient_onto_residue(ref_res, alignment_vector)
            # print("post_align")
            aligned_rots.append(aligned_rot)

        return raw_rots, aligned_rots, alignment_vector

    def generate_aligned_rotamers_GXG(self, GXG_pose, ref_res=None):
        alignment_atoms_dict={}
        alignment_atoms_dict['D'] = ["CG", "OD1", "OD2"]
        alignment_atoms_dict['N'] = ["CG", "OD1", "ND2"]
        alignment_atoms_dict['E'] = ["CD", "OE1", "OE2"]
        alignment_atoms_dict['Q'] = ["CD", "OE1", "NE2"]
        alignment_atoms_dict['R'] = ["CZ", "NH1", "NH2"]
        alignment_atoms_dict['K'] = ["CD", "CE", "NZ"]
        alignment_atoms_dict['W'] = ["CD1", "NE1", "CE2"]
        alignment_atoms_dict['H'] = ["ND1", "CE1", "NE2"]
        alignment_atoms_dict['Y'] = ["CG","CE1","CE2"]

        SC_idx = 2
        ori_res = GXG_pose.residue(SC_idx).clone()
        if not ref_res:
            ref_res = ori_res.clone()
        if not ori_res.name1() in alignment_atoms_dict.keys(): return None

        alignment_atoms = alignment_atoms_dict[ori_res.name1()]
        alignment_vector = pyrosetta.rosetta.utility.vector1_std_pair_std_string_std_string_t()
        for atom in alignment_atoms:
            alignment_vector.append( (atom, atom) )

        ## generate raw rotamers, not aligned yet
        raw_rots=self.get_rots_GXG(GXG_pose)
        print(f"Number of inverse rotamers used in hashing for res {ori_res.name3()}: {len(raw_rots)}")

        aligned_rots=[]
        for rot in raw_rots:
            aligned_rot = rot.clone()
            # print("pre_align")
            aligned_rot.orient_onto_residue(ref_res, alignment_vector)
            # print("post_align")
            aligned_rots.append(aligned_rot)

        return raw_rots, aligned_rots, alignment_vector

    def generate_aligned_GXG(self, GXG_pose):
        alignment_atoms_dict={}
        alignment_atoms_dict['D'] = ["CG", "OD1", "OD2"]
        alignment_atoms_dict['N'] = ["CG", "OD1", "ND2"]
        alignment_atoms_dict['E'] = ["CD", "OE1", "OE2"]
        alignment_atoms_dict['Q'] = ["CD", "OE1", "NE2"]
        alignment_atoms_dict['R'] = ["CZ", "NH1", "NH2"]
        alignment_atoms_dict['K'] = ["CD",  "CE",  "NZ"]
        alignment_atoms_dict['W'] = ["CD1", "NE1", "CE2"]
        alignment_atoms_dict['H'] = ["ND1", "CE1", "NE2"]
        alignment_atoms_dict['Y'] = ["CG",  "CE1", "CE2"]

        SC_idx = 2

        ## generate raw rotamers, not aligned yet
        raw_rots=self.get_rots_GXG(GXG_pose)
        raw_GXG_list = []
        aligned_GXG_list=[]
        for ii, each_raw in enumerate(raw_rots):
            raw_GXG_pose = GXG_pose.clone()
            raw_GXG_pose.replace_residue(SC_idx, each_raw, True)
            # raw_GXG_pose.dump_pdb(f"raw_GXG_{ii}.pdb")
            raw_GXG_list.append(raw_GXG_pose.clone())

            aligned_GXG = raw_GXG_pose.clone()
            residues_mobile = [SC_idx]
            atoms_mobile    = alignment_atoms_dict[GXG_pose.residue(SC_idx).name1()]
            residues_ref    = residues_mobile
            atoms_ref       = atoms_mobile
            coordinates_mobile = get_xyz(aligned_GXG, residues_mobile, atoms_mobile)
            coordinates_ref    = get_xyz(GXG_pose,    residues_ref,    atoms_ref)

            rot_M, trans_V = get_rotate_translate(np.matrix(coordinates_mobile), np.matrix(coordinates_ref))

            old_mobile_pose_coordinates = get_xyz(aligned_GXG)
            coordinates_output = transform_query_coordinates(np.matrix(old_mobile_pose_coordinates), rot_M, trans_V)
            set_xyz(aligned_GXG, coordinates_output)
            aligned_GXG_list.append(aligned_GXG)

        return raw_GXG_list, aligned_GXG_list

    @staticmethod
    def get_alignment_dict():
        ## use three heavy atoms for reverse rotamer alignment
        ## not always the last three heavy atoms, for example, Ser, His, Pro , Cys, and Trp
        alignment_atoms_dict=defaultdict(lambda: (["N", "CA", "O"], ["N", "CA", "O"]))
        alignment_atoms_dict['N'] = ( ["CG", "OD1", "ND2"], ["N", "CA", "O"] )
        alignment_atoms_dict['S'] = ( ["CA", "CB", "OG"], ["CA", "CB", "OG"] )
        alignment_atoms_dict['E'] = ( ["CD", "OE1", "OE2"], ["N", "CA", "O"] )
        alignment_atoms_dict['R'] = ( ["CZ", "NH1", "NH2"], ["N", "CA", "O"] )
        alignment_atoms_dict['K'] = ( ["CD", "CE", "NZ"], ["N", "CA", "O"] )
        alignment_atoms_dict['W'] = ( ["CD1", "NE1", "CE2"], ["N", "CA", "O"] )
        alignment_atoms_dict['H'] = ( ["ND1", "CE1", "NE2"], ["N", "CA", "O"] )
        alignment_atoms_dict['T'] = ( ["CA", "CB", "OG1"], ["CA", "CB", "OG1"] )
        # alignment_atoms_dict['Y'] = ( ["CE2", "CZ","OH"], ["N", "CA", "O"] )
        alignment_atoms_dict['Y'] = ( ["CG","CE1","CE2"], ["N", "CA", "O"] )
        alignment_atoms_dict['D'] = ( ["CG", "OD1", "OD2"], ["N", "CA", "O"] )
        alignment_atoms_dict['P'] = ( ["CA", "C", "O"], ["N", "CA", "O"] )
        alignment_atoms_dict['Q'] = ( ["CD","OE1","NE2"], ["N", "CA", "O"] )
        alignment_atoms_dict['C'] = ( ["CA", "CB", "SG"], ["N", "CA", "O"] )
        alignment_atoms_dict['F'] = ( ["CG","CE1","CE2"], ["N", "CA", "O"] )

        # align_dict['W']=["CD2","CZ2","CZ3"]
        # align_dict['Y']=["CG","CE1","CE2"]
        # align_dict['F']=["CG","CE1","CE2"]
        # align_dict['H']=["CG","ND1","NE2"]
        return alignment_atoms_dict

    @staticmethod
    def get_SCRR_enum():
        SC_choices = ["D", "N", "E", "Q", "K", "R", "H", "S", "T", "W", "Y"]
        rayray_choices = [ ("NH", "CO"), ("CO", "NH"), ("NH", "NH"), ("CO", "CO"), ("NH", ""), ("CO", "") ]
        potential_SCRR_list = [ f"{each_SCRR[0]}{each_SCRR[1][0]}{each_SCRR[1][1]}" for each_SCRR in product(SC_choices, rayray_choices) ]
        assert(len(potential_SCRR_list) <= 255)
        ## SCRR: sidechain-rayrary
        SCRR_enum = Enum("SCRR", potential_SCRR_list)
        return SCRR_enum

    @staticmethod
    def get_hash_entry_dtype():
        return np.dtype([("hashkey", np.uint64, 1),
                         ("SCRR",  np.uint8, 1),
                         ("chisSC", np.int8, (4,))])

    @staticmethod
    def check_hbond_types(hbset, aa_trio, res_idx):
        hbonds = hbset.residue_hbonds( res_idx )
        hbond_type = []
        for hbond in hbonds:
            don_idx = hbond.don_res()
            don_atom = hbond.don_hatm() ## mean the H, not heavyatom
            don_res = aa_trio.residue(don_idx)
            don_atom_name = don_res.atom_name(don_atom)
            don_name3 = don_res.name3()

            acc_idx = hbond.acc_res()
            acc_atom = hbond.acc_atm()
            acc_res = aa_trio.residue(acc_idx)
            acc_atom_name = acc_res.atom_name(acc_atom)
            acc_name3 = acc_res.name3()

            ## trick: use strip to remove the white space around atom names
            if don_idx == res_idx and hbond.don_hatm_is_backbone() \
                                  and don_atom_name.strip() == "H" \
                                  and not hbond.acc_atm_is_backbone():

                hbond_type.append("NH")

            if acc_idx == res_idx and hbond.acc_atm_is_backbone() \
                                  and acc_atom_name.strip() == "O" \
                                  and not hbond.don_hatm_is_backbone():

                hbond_type.append("CO")

        return list(set(hbond_type))

    ## find out the types of the two rays. For example: "CO"+"NH" or "NH"+"NH"
    @staticmethod
    def check_ray_type(input_pose, SC_idx, BB_idx_list):
        scorefxn = pyrosetta.create_score_function("ref2015")
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)
        scorefxn(input_pose)

        hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
        pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(input_pose, calculate_derivative=False, hbond_set=hbset)

        if BB_idx_list[0] == BB_idx_list[1]:
            Aray_str = "NH"
            Bray_str = "CO"

        else:
            Nterm_type = aa_trio_hash_base.check_hbond_types(hbset, input_pose, BB_idx_list[0] )
            Cterm_type = aa_trio_hash_base.check_hbond_types(hbset, input_pose, BB_idx_list[1] )
            try:
                assert (len(Nterm_type) == 1 and len(Cterm_type) == 1)
            except:
                print("too much hbond?")
                return None

            Aray_str = Nterm_type[0]
            Bray_str = Cterm_type[0]

        return Aray_str, Bray_str




























