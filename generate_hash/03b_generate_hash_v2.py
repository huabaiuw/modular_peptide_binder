#!/usr/bin/env python
import pickle
import hash_helpers
import pyrosetta
from glob import glob
import os,sys,re
from collections import defaultdict
from pyrosetta.rosetta.core.scoring import fa_rep, fa_dun, fa_atr
# from scipy.spatial.transform import Rotation as R
import numpy as np
import transformations
import rif.hash
from rif.geom.ray_hash import RayRay10dHash
from rif.geom import Ray
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3
from random import shuffle
from itertools import product

def create_dir(out_dir):
    if not os.path.isdir(out_dir):
        try:
            os.mkdir(out_dir)
        except Exception:
            pass

def generate_random_M(angle_sd, xyz_sd):
    dphi = np.random.normal(0, angle_sd/180*np.pi)
    dtheta = np.random.normal(0, angle_sd/180*np.pi)
    dpsi = np.random.normal(0, angle_sd/180*np.pi)
    dx = np.random.normal(0, xyz_sd)
    dy = np.random.normal(0, xyz_sd)
    dz = np.random.normal(0, xyz_sd)
    return transformations.compose_matrix(angles=[dphi,dtheta,dpsi], translate=[dx, dy, dz])

def get_new_xyz(input_xyz, M):
    input_xyz_addone = np.ones((input_xyz.shape[0],input_xyz.shape[1]+1))
    input_xyz_addone[:,:-1] = input_xyz
    output_xyz_addone = np.dot(M, input_xyz_addone.transpose())
    output_xyz = output_xyz_addone[:3,:].transpose()

    return output_xyz

def get_res_xyz(input_res, atom_names = "all"):
    output_xyz_list = []
    if atom_names == "all":
        for ai in range(1, input_res.natoms()+1):
            xyz = input_res.xyz(ai)
            output_xyz_list.append([xyz[0], xyz[1], xyz[2]])
    elif atom_names == "heavy":
        for ai in range(1, input_res.nheavyatoms()+1):
            xyz = input_res.xyz(ai)
            output_xyz_list.append([xyz[0], xyz[1], xyz[2]])

    elif atom_names == "sc":
        heavy_idx_set = set([x for x in range(1, input_res.nheavyatoms()+1)])
        bb_idx_set = set(input_res.all_bb_atoms())
        res_sc_heavy = list(heavy_idx_set - bb_idx_set)
        for ai in res_sc_heavy:
            xyz = input_res.xyz(ai)
            output_xyz_list.append([xyz[0], xyz[1], xyz[2]])
    else:
        for an in atom_names:
            ## use atom_name in atom_names could be better
            ## atom_name = ref_pose.residue(ir).atom_name(ia).strip()
            xyz = input_res.xyz(an)
            output_xyz_list.append([xyz[0], xyz[1], xyz[2]])
    return np.array(output_xyz_list)

def set_res_xyz(input_res, new_coords):
    idx = 0
    for ia in range(1, input_res.natoms() + 1):
        vector = V3(new_coords[idx][0], new_coords[idx][1], new_coords[idx][2])
        input_res.set_xyz(ia, vector)
        idx += 1
    return input_res



def generate_new_res(input_res, M):
    output_res = input_res.clone()
    input_xyz = get_res_xyz(input_res)
    output_xyz = get_new_xyz(input_xyz, M)
    set_res_xyz(output_res, output_xyz)
    return output_res

def get_stub(res, stub_atoms = ["N", "CA", "C"]):
    return pyrosetta.rosetta.core.kinematics.Stub( V3(res.xyz(stub_atoms[0])),
                                                   V3(res.xyz(stub_atoms[1])),
                                                   V3(res.xyz(stub_atoms[2])) )

def get_hash_key(hash_function, ref_res, ray_A_res, ray_B_res,
                 stub_atoms = ["N", "CA", "C"],
                 ray_A_atoms = ["N", "H"],
                 ray_B_atoms = ["C", "O"] ):

    ref_stub = get_stub(ref_res, stub_atoms)
    ray_A_start = ref_stub.global2local( ray_A_res.xyz( ray_A_atoms[0] ) )
    ray_A_end   = ref_stub.global2local( ray_A_res.xyz( ray_A_atoms[1] ) )
    ray_B_start = ref_stub.global2local( ray_B_res.xyz( ray_B_atoms[0] ) )
    ray_B_end   = ref_stub.global2local( ray_B_res.xyz( ray_B_atoms[1] ) )
    ray_A = Ray( orig=ray_A_start, dirn=( ray_A_end - ray_A_start ) )
    ray_B = Ray( orig=ray_B_start, dirn=( ray_B_end - ray_B_start ) )
    rayray_hash_key = hash_function.get_key( ray_A, ray_B )
    return rayray_hash_key


def get_stub_hash_key(hash_function, ref_res, ref_stub_atoms, key_res, key_stub_atoms= ["N", "CA", "C"]):
    ref_stub = get_stub(ref_res, ref_stub_atoms)
    stub_xyz_1 = ref_stub.global2local(key_res.xyz(key_stub_atoms[0]))
    stub_xyz_2 = ref_stub.global2local(key_res.xyz(key_stub_atoms[1]))
    stub_xyz_3 = ref_stub.global2local(key_res.xyz(key_stub_atoms[2]))
    key_stub = pyrosetta.rosetta.core.kinematics.Stub(stub_xyz_1, stub_xyz_2, stub_xyz_3)

    return hash_function.get_key(key_stub)

def generate_new_pose_by_M(input_pose, M):
    input_xyz = hash_helpers.get_xyz(input_pose)
    input_xyz_addone = np.ones((input_xyz.shape[0],input_xyz.shape[1]+1))
    input_xyz_addone[:,:-1] = input_xyz
    output_xyz_addone = np.dot(M, input_xyz_addone.transpose())
    output_xyz = output_xyz_addone[:3,:].transpose()
    output_pose = input_pose.clone()
    hash_helpers.set_xyz(output_pose, output_xyz)
    return output_pose

def align_pose_to_ring(ref_pose, ref_idx, query_pose, query_idx):
    ring_stub_dict = {}
    ring_stub_dict["W"] = ["CG", "CE3", "CZ2"]
    ring_stub_dict["Y"] = ["CG", "CE1", "CE2"]
    ring_stub_dict["F"] = ["CG", "CE1", "CE2"]
    ring_stub_dict["H"] = ["CG", "CE1", "NE2"]
    ring_stub_dict["R"] = ["CZ", "NH1", "NH2"]
    ring_stub_dict["K"] = ["CE", "NZ", "CD"]

    ref_rot   = ref_pose.residue(ref_idx)
    query_rot = query_pose.residue(query_idx)

    assert ref_rot.name1() == query_rot.name1()
    AA_id = query_rot.name1()
    atoms_to_align    = ring_stub_dict[AA_id]

    ## extract the xyz of the three atoms
    coordinates_mobile = get_res_xyz(query_rot, atoms_to_align)
    coordinates_ref    = get_res_xyz(ref_rot, atoms_to_align)

    assert(coordinates_mobile.shape == coordinates_ref.shape)

    M = transformations.affine_matrix_from_points(coordinates_mobile.transpose(),
                                                  coordinates_ref.transpose(),
                                                  shear=False,
                                                  scale=False)

    out_pose = generate_new_pose_by_M(query_pose, M)
    return out_pose


def align_rot_to_ring(ref_rot, query_rot):
    ring_stub_dict = {}
    ring_stub_dict["W"] = ["CG", "CE3", "CZ2"]
    ring_stub_dict["Y"] = ["CG", "CE1", "CE2"]
    ring_stub_dict["F"] = ["CG", "CE1", "CE2"]
    ring_stub_dict["H"] = ["CG", "CE1", "NE2"]
    ring_stub_dict["R"] = ["CZ", "NH1", "NH2"]
    ring_stub_dict["K"] = ["CE", "NZ", "CD"]

    assert ref_rot.name1() == query_rot.name1()
    AA_id = query_rot.name1()
    atoms_to_align    = ring_stub_dict[AA_id]

    ## extract the xyz of the three atoms
    coordinates_mobile = get_res_xyz(query_rot, atoms_to_align)
    coordinates_ref    = get_res_xyz(ref_rot, atoms_to_align)

    assert(coordinates_mobile.shape == coordinates_ref.shape)

    M = transformations.affine_matrix_from_points(coordinates_mobile.transpose(),
                                                  coordinates_ref.transpose(),
                                                  shear=False,
                                                  scale=False)

    input_xyz = get_res_xyz(query_rot)
    input_xyz_addone = np.ones((input_xyz.shape[0],input_xyz.shape[1]+1))
    input_xyz_addone[:,:-1] = input_xyz
    output_xyz_addone = np.dot(M, input_xyz_addone.transpose())
    output_xyz = output_xyz_addone[:3,:].transpose()

    output_res = query_rot.clone()
    set_res_xyz(output_res, output_xyz)
    return output_res


def align_rot_to_polar(ref_rot, query_rot):
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

    assert ref_rot.name1() == query_rot.name1()
    AA_id = query_rot.name1()
    atoms_to_align    = alignment_atoms_dict[AA_id]

    ## extract the xyz of the three atoms
    coordinates_mobile = get_res_xyz(query_rot, atoms_to_align)
    coordinates_ref    = get_res_xyz(ref_rot, atoms_to_align)

    assert(coordinates_mobile.shape == coordinates_ref.shape)

    M = transformations.affine_matrix_from_points(coordinates_mobile.transpose(),
                                                  coordinates_ref.transpose(),
                                                  shear=False,
                                                  scale=False)

    input_xyz = get_res_xyz(query_rot)
    input_xyz_addone = np.ones((input_xyz.shape[0],input_xyz.shape[1]+1))
    input_xyz_addone[:,:-1] = input_xyz
    output_xyz_addone = np.dot(M, input_xyz_addone.transpose())
    output_xyz = output_xyz_addone[:3,:].transpose()

    output_res = query_rot.clone()
    set_res_xyz(output_res, output_xyz)
    return output_res


def simple_mutate_res(input_pose, res_idx, mutant_AA):
    new_aa_name=pyrosetta.rosetta.core.chemical.name_from_aa(pyrosetta.rosetta.core.chemical.aa_from_oneletter_code( mutant_AA ))
    restype_set=input_pose.residue_type_set_for_pose()
    new_res = pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue(rsd_type=restype_set.name_map(new_aa_name),
                                                                         current_rsd=input_pose.residue(res_idx),
                                                                         conformation=input_pose.conformation(),
                                                                         preserve_c_beta=False)

    input_pose.replace_residue( seqpos=res_idx, new_rsd_in=new_res, orient_backbone=False )

    hash_helpers.check_and_remove_variant(input_pose, res_idx)


def extract_weighted_score(input_pose, res_idx, scorefxn, score_term):
    input_pose.energies().clear()
    scorefxn(input_pose)
    input_energies = input_pose.energies()
    raw_value = input_energies.residue_total_energies(res_idx)[score_term]
    weighted_value = raw_value * scorefxn.weights()[score_term]
    return weighted_value


if __name__ == "__main__":
    pwd = os.getcwd()
    pwd = re.sub("/mnt", "", pwd)
    pwd = re.sub("/laobai/digs", "", pwd)
    scratch_dir = "/home/huabai/net_scratch/DBs_scratch"

    DBs_dir = f"{pwd}/DBs_raw"
    if not os.path.exists(DBs_dir):
        os.mkdir(DBs_dir)

    test_dir = f"{pwd}/local_test"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    ## original rots dictionary:
    ## rot_idx: (AA_id, chi_array)
    rots_pickle = "./catpi_rots_dict.pickle"
    with open(rots_pickle, "rb") as f:
         rots_dict = pickle.load(f)

    ## convert dict format
    ## key:  AA_id
    ## value: dict: {rot_idx: chi_array}
    AA_rot_dict = defaultdict(lambda: {})
    for rot_idx, rot_info in rots_dict.items():
        AA_id     = rot_info[0]
        chi_array = rot_info[1]
        AA_rot_dict[AA_id][rot_idx] = chi_array

    pyrosetta.init("-mute all")

    input_pdb    = sys.argv[1]
    input_pair_id = sys.argv[2]
    N_trial = int(sys.argv[3])

    print(input_pdb)
    id_pattern = re.compile("key_pair_([A-Z]{2,})_")
    catpi_id = id_pattern.findall(input_pdb)[0]
    assert catpi_id == input_pair_id
    print(catpi_id)


    cat_idx = 1
    pi_idx  = 2
    this_pose = pyrosetta.pose_from_file(input_pdb)

    cat_res = this_pose.residue(cat_idx).clone()
    pi_res  = this_pose.residue(pi_idx).clone()
    pair_pose = pyrosetta.Pose()
    pair_pose.append_residue_by_jump(cat_res, 1)
    pair_pose.append_residue_by_jump(pi_res, 1)

    # pair_pose.dump_pdb("test.pdb")
    scorefxn = pyrosetta.create_score_function("ref2015")
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)

    base_cat_fa_rep = extract_weighted_score(pair_pose, cat_idx, scorefxn, fa_rep)
    base_cat_fa_atr = extract_weighted_score(pair_pose, cat_idx, scorefxn, fa_atr)
    base_cat_fa_dun = extract_weighted_score(pair_pose, cat_idx, scorefxn, fa_dun)
    base_cat_rep_atr = base_cat_fa_rep + base_cat_fa_atr
    # print(base_cat_fa_rep)
    # print(base_cat_fa_atr)
    # print(base_cat_rep_atr)
    if base_cat_rep_atr > 0: sys.exit()

    base_pi_fa_rep = extract_weighted_score(pair_pose, pi_idx, scorefxn, fa_rep)
    base_pi_fa_atr = extract_weighted_score(pair_pose, pi_idx, scorefxn, fa_atr)
    base_pi_fa_dun = extract_weighted_score(pair_pose, pi_idx, scorefxn, fa_dun)
    base_pi_rep_atr = base_pi_fa_rep + base_pi_fa_atr
    # print(base_pi_fa_rep)
    # print(base_pi_fa_atr)
    # print(base_pi_rep_atr)
    if base_pi_rep_atr > 0: sys.exit()

    cat_name = pair_pose.residue(1).name1()
    pi_name  = pair_pose.residue(2).name1()
    catpi_id_v2 = f"{cat_name}{pi_name}"
    assert catpi_id_v2 == catpi_id


    sub_rots_cat_dict  = AA_rot_dict[cat_name]
    sub_rots_cat_count = len(sub_rots_cat_dict)
    print(sub_rots_cat_count)

    sub_rots_pi_dict  = AA_rot_dict[pi_name]
    sub_rots_pi_count = len(sub_rots_pi_dict)
    print(sub_rots_pi_count)

    base_AB_dist, base_AB_plane_angle, base_AB_shift_angle, _ = hash_helpers.survey_catpi_angle(pair_pose, cat_idx, cat_name, pi_idx, pi_name)

    print(f"{base_AB_dist:.2f}")
    print(f"{base_AB_plane_angle:.1f}")
    print(f"{base_AB_shift_angle:.1f}")

    if base_AB_dist > 5: sys.exit()
    if base_AB_plane_angle > 10: sys.exit()
    if base_AB_shift_angle > 30: sys.exit()

    cart_resl  = 1
    ori_resl   = 10
    cart_bound = 512
    hash_function = rif.hash.RosettaStubHash(cart_resl, ori_resl, cart_bound)

    ## cat is the ref_res
    ## pi is the query_res
    ref_res = pair_pose.residue(cat_idx)
    ref_stub_atoms = ["N", "CA", "C"]
    key_res = pair_pose.residue(pi_idx)
    key_stub_atoms = ["N", "CA", "C"]

    base_key = get_stub_hash_key(hash_function, ref_res, ref_stub_atoms, key_res, key_stub_atoms)
    print(base_key)

    out_dtype = np.dtype([("key",       "u8"),
                          ("chi_idx_cat", "i2"),
                          ("chi_idx_pi", "i2")])

    # raw_size = N_trial * sub_rots_count
    out_dat = f"{DBs_dir}/raw_key_chi_idx_{catpi_id}_{np.random.randint(1e6, 1e7)}.dat"
    while os.path.isfile(out_dat):
        out_dat = f"{DBs_dir}/raw_key_chi_idx_{catpi_id}_{np.random.randint(1e6, 1e7)}.dat"
    out_array = np.memmap(out_dat, mode="w+", dtype=out_dtype, shape=(1,))

    base_sc_cat = pair_pose.residue(cat_idx).clone()
    base_sc_pi  = pair_pose.residue(pi_idx).clone()

    rot_idx_list_cat = [x for x,y in sub_rots_cat_dict.items()]
    shuffle(rot_idx_list_cat)

    rot_idx_list_pi = [x for x,y in sub_rots_pi_dict.items()]
    shuffle(rot_idx_list_pi)

    key_count = 0

    for ii in range(N_trial):
        angle_sd = 60
        xyz_sd = 1
        random_M = generate_random_M(angle_sd, xyz_sd)

        if ii == 0:
            random_M = np.identity(4)

        part_cat = pyrosetta.Pose(pair_pose.clone(), 1, 1)
        part_pi = pyrosetta.Pose(pair_pose.clone(), 2, 2)

        aligned_cat = align_pose_to_ring(pair_pose, 1, part_cat, 1)
        aligned_pi = align_pose_to_ring(pair_pose, 2, part_pi, 1)

        jitter_cat = generate_new_pose_by_M(aligned_cat, random_M)
        jitter_pi = aligned_pi.clone()

        jitter_pose = jitter_cat.clone()
        jitter_pose.append_pose_by_jump(jitter_pi, 1)
        jitter_pose.energies().clear()
        scorefxn(jitter_pose)

        this_cat_fa_rep = extract_weighted_score(jitter_pose, cat_idx, scorefxn, fa_rep)
        this_cat_fa_atr = extract_weighted_score(jitter_pose, cat_idx, scorefxn, fa_atr)
        this_cat_fa_dun = extract_weighted_score(jitter_pose, cat_idx, scorefxn, fa_dun)
        this_cat_rep_atr = this_cat_fa_rep + this_cat_fa_atr

        this_pi_fa_rep = extract_weighted_score(jitter_pose, pi_idx, scorefxn, fa_rep)
        this_pi_fa_atr = extract_weighted_score(jitter_pose, pi_idx, scorefxn, fa_atr)
        this_pi_fa_dun = extract_weighted_score(jitter_pose, pi_idx, scorefxn, fa_dun)
        this_pi_rep_atr = this_pi_fa_rep + this_pi_fa_atr

        if this_cat_rep_atr > base_cat_rep_atr + 0.1: continue
        if this_pi_rep_atr > base_pi_rep_atr + 0.1: continue

        total_fa_rep = jitter_pose.energies().total_energies()[fa_rep]
        total_fa_atr = jitter_pose.energies().total_energies()[fa_atr]
        total_rep_atr  = total_fa_rep * scorefxn.weights()[fa_rep] +\
                         total_fa_atr * scorefxn.weights()[fa_atr]
        if total_rep_atr > 0: continue
        jitter_AB_dist, jitter_AB_plane_angle, jitter_AB_shift_angle, _ = hash_helpers.survey_catpi_angle(jitter_pose, cat_idx, cat_name, pi_idx, pi_name)

        if jitter_AB_dist > 5: continue
        if jitter_AB_plane_angle > 10: continue
        if jitter_AB_shift_angle > 30: continue

        rot_idx_pairs = [(x,y) for x, y in product(rot_idx_list_cat, rot_idx_list_pi)]
        shuffle(rot_idx_pairs)


        for rot_idx_cat, rot_idx_pi in rot_idx_pairs:

            part_cat = pyrosetta.Pose(jitter_pose.clone(), 1, 1)
            part_pi = pyrosetta.Pose(jitter_pose.clone(), 2, 2)

            chi_array_cat = sub_rots_cat_dict[rot_idx_cat]
            chi_array_pi = sub_rots_pi_dict[rot_idx_pi]

            for jj in range(1, part_cat.residue(1).nchi()+1):
                part_cat.residue(1).set_chi(jj, chi_array_cat[jj-1])

            aligned_cat = align_pose_to_ring(pair_pose, 1, part_cat, 1)
            for jj in range(1, part_pi.residue(1).nchi()+1):
                part_pi.residue(1).set_chi(jj, chi_array_pi[jj-1])
            aligned_pi = align_pose_to_ring(pair_pose, 2, part_pi, 1)

            swap_rot_pose = aligned_cat.clone()
            swap_rot_pose.append_pose_by_jump(aligned_pi.clone(), 1)

            swap_rot_pose.energies().clear()
            scorefxn(swap_rot_pose)

            this_cat_fa_rep = extract_weighted_score(swap_rot_pose, cat_idx, scorefxn, fa_rep)
            this_cat_fa_atr = extract_weighted_score(swap_rot_pose, cat_idx, scorefxn, fa_atr)
            this_cat_fa_dun = extract_weighted_score(swap_rot_pose, cat_idx, scorefxn, fa_dun)
            this_cat_rep_atr = this_cat_fa_rep + this_cat_fa_atr

            this_pi_fa_rep = extract_weighted_score(swap_rot_pose, pi_idx, scorefxn, fa_rep)
            this_pi_fa_atr = extract_weighted_score(swap_rot_pose, pi_idx, scorefxn, fa_atr)
            this_pi_fa_dun = extract_weighted_score(swap_rot_pose, pi_idx, scorefxn, fa_dun)
            this_pi_rep_atr = this_pi_fa_rep + this_pi_fa_atr

            if this_cat_rep_atr > base_cat_rep_atr + 0.1: continue
            if this_pi_rep_atr > base_pi_rep_atr + 0.1: continue
            total_fa_rep = swap_rot_pose.energies().total_energies()[fa_rep]
            total_fa_atr = swap_rot_pose.energies().total_energies()[fa_atr]
            total_rep_atr  = total_fa_rep * scorefxn.weights()[fa_rep] +\
                             total_fa_atr * scorefxn.weights()[fa_atr]

            if total_rep_atr > 0: continue

            ref_res = swap_rot_pose.residue(cat_idx)
            ref_stub_atoms = ["N", "CA", "C"]
            key_res = swap_rot_pose.residue(pi_idx)
            key_stub_atoms = ["N", "CA", "C"]
            this_key = get_stub_hash_key(hash_function, ref_res, ref_stub_atoms, key_res, key_stub_atoms)
            print(this_key)
            each_entry = np.zeros(1, dtype=out_dtype)
            each_entry["key"]     = this_key
            each_entry["chi_idx_cat"] = rot_idx_cat
            each_entry["chi_idx_pi"] = rot_idx_pi

            adding_memmap_array = np.memmap(out_dat, mode="r+",
                                            dtype=out_dtype,
                                            shape=(1,),
                                            offset = key_count*out_dtype.itemsize)

            adding_memmap_array[0] = each_entry[0]
            key_count += 1

            # swap_rot_pose.dump_pdb(f"{test_dir}/{key_count}_{catpi_id}_{this_key}.pdb")
