#!/usr/bin/env python
import json
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
from rif.hash import RosettaStubTwoTorsionHash
from rif.geom.ray_hash import RayRay10dHash
from rif.geom import Ray
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3
from random import shuffle
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist



def generate_new_pose_by_M(input_pose, M):
    input_xyz = hash_helpers.get_xyz(input_pose)
    input_xyz_addone = np.ones((input_xyz.shape[0],input_xyz.shape[1]+1))
    input_xyz_addone[:,:-1] = input_xyz
    output_xyz_addone = np.dot(M, input_xyz_addone.transpose())
    output_xyz = output_xyz_addone[:3,:].transpose()
    output_pose = input_pose.clone()
    hash_helpers.set_xyz(output_pose, output_xyz)
    return output_pose

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

def get_rr_rash_key(hashfxn,
                 ref_res, ray_A_res, ray_B_res,
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
    rayray_hash_key = hashfxn.get_key( ray_A, ray_B )
    return rayray_hash_key


def get_stubtor_key(hashfxn,ref_res,query_res,ref_atoms,query_atoms,query_phi,query_psi):
    ref_stub = get_stub(ref_res, ref_atoms)

    query_local_1 = ref_stub.global2local( query_res.xyz( query_atoms[0] ) )
    query_local_2 = ref_stub.global2local( query_res.xyz( query_atoms[1] ) )
    query_local_3 = ref_stub.global2local( query_res.xyz( query_atoms[2] ) )
    query_stub = pyrosetta.rosetta.core.kinematics.Stub( query_local_1, query_local_2, query_local_3 )

    return hashfxn.get_key(query_stub, query_phi, query_psi)

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


def check_ray_type(input_pose, SC_idx, BB_idx):
    scorefxn = pyrosetta.create_score_function("ref2015")
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)
    try:
        scorefxn(input_pose)
    except Exception:
        return []

    hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(input_pose, calculate_derivative=False, hbond_set=hbset)
    hbond_list = hash_helpers.check_hbond_types(hbset, input_pose, BB_idx)
    return hbond_list


def align_to_bb(input_pose, key_idx):
    alignment_atoms_dict={}
    alignment_atoms_dict['D'] = ["CA", "N", "C"]
    alignment_atoms_dict['N'] = ["CA", "N", "C"]
    alignment_atoms_dict['E'] = ["CA", "N", "C"]
    alignment_atoms_dict['Q'] = ["CA", "N", "C"]
    alignment_atoms_dict['R'] = ["CA", "N", "C"]
    alignment_atoms_dict['K'] = ["CA", "N", "C"]
    alignment_atoms_dict['W'] = ["CA", "N", "C"]
    alignment_atoms_dict['H'] = ["CA", "N", "C"]
    alignment_atoms_dict['Y'] = ["CA", "N", "C"]

    key_res = input_pose.residue(key_idx)
    AA_id = key_res.name1()
    atoms_to_align    = alignment_atoms_dict[AA_id]

    ## extract the xyz of the three atoms
    coordinates_mobile = get_res_xyz(key_res, atoms_to_align)

    xyz_mobile_1 = coordinates_mobile[0]

    ## new xyz for coordinates_mobile[0]
    x1 = 0
    y1 = 0
    z1 = 0

    vector12 = coordinates_mobile[1] - coordinates_mobile[0]
    vector13 = coordinates_mobile[2] - coordinates_mobile[0]

    angle23 = transformations.angle_between_vectors( vector12, vector13 )
    angle23 = angle23 / 2

    ## new xyz for coordinates_mobile[1]
    x2 = - np.linalg.norm(vector12) * np.sin(angle23)
    y2 = 0
    z2 = - np.linalg.norm(vector12) * np.cos(angle23)

    ## new xyz for coordinates_mobile[2]
    x3 = np.linalg.norm(vector12) * np.sin(angle23)
    y3 = 0
    z3 = - np.linalg.norm(vector12) * np.cos(angle23)

    ## the new xyz for CD or CG is [0,0,0]
    coordinates_ref = np.array([[x1, y1, z1],[x2, y2, z2],[x3, y3, z3]])
    assert(coordinates_mobile.shape == coordinates_ref.shape)

    M = transformations.affine_matrix_from_points(coordinates_mobile.transpose(),
                                                  coordinates_ref.transpose(),
                                                  shear=False,
                                                  scale=False)

    input_xyz = hash_helpers.get_xyz(input_pose)
    input_xyz_addone = np.ones((input_xyz.shape[0],input_xyz.shape[1]+1))
    input_xyz_addone[:,:-1] = input_xyz

    output_xyz_addone = np.dot(M, input_xyz_addone.transpose())
    output_xyz = output_xyz_addone[:3,:].transpose()

    output_pose = input_pose.clone()
    hash_helpers.set_xyz(output_pose, output_xyz)
    return output_pose

def extract_weighted_score(input_pose, res_idx, scorefxn, score_term):
    input_pose.energies().clear()
    scorefxn(input_pose)
    input_energies = input_pose.energies()
    raw_value = input_energies.residue_total_energies(res_idx)[score_term]
    weighted_value = raw_value * scorefxn.weights()[score_term]
    return weighted_value


def align_to_ref_bb(ref_pose, ref_idx, query_pose, query_idx, bb_atoms =["N", "CA", "C"]):

    ## extract the xyz of the three atoms
    coordinates_mobile = hash_helpers.get_xyz(query_pose, residue_range=[query_idx], atom_names=bb_atoms)
    coordinates_ref    = hash_helpers.get_xyz(ref_pose,   residue_range=[ref_idx],   atom_names=bb_atoms)
    assert(coordinates_mobile.shape == coordinates_ref.shape)

    M = transformations.affine_matrix_from_points(coordinates_mobile.transpose(),
                                                  coordinates_ref.transpose(),
                                                  shear=False,
                                                  scale=False)

    input_xyz = hash_helpers.get_xyz(query_pose)
    input_xyz_addone = np.ones((input_xyz.shape[0],input_xyz.shape[1]+1))
    input_xyz_addone[:,:-1] = input_xyz
    output_xyz_addone = np.dot(M, input_xyz_addone.transpose())
    output_xyz = output_xyz_addone[:3,:].transpose()

    output_pose = query_pose.clone()
    hash_helpers.set_xyz(output_pose, output_xyz)
    return output_pose


def load_rots_info():
    with open("rot_idx_allocation.json", "r") as f:
        allocation_dict = json.load(f)

    ## if there are exsiting rotamers information
    ## load them
    exsiting_json = "rotamers.json"
    if os.path.isfile(exsiting_json):
        with open(exsiting_json, "r") as f:
            raw_rots_dict = json.load(f)

        ## change the rot_idx from str to int
        old_rots_dict = defaultdict(lambda: {})
        for rot_AA, chis_dict in raw_rots_dict.items():
            for rot_idx, chi_set in chis_dict.items():
                old_rots_dict[rot_AA][int(rot_idx)] = chi_set
    else:
        old_rots_dict = defaultdict(lambda:{})

    ## double theck the old rots are compatible to the allocated AA
    for rot_AA, chis_dict in old_rots_dict.items():
        for rot_idx in chis_dict.keys():
            assert rot_idx in range(*allocation_dict[rot_AA])

    return old_rots_dict


def generate_aligned_rots(input_pose, AA_rot_dict):
    sc_idx = 2
    sc_AA = input_pose.residue(sc_idx).name1()
    ## just extract the rotamer dict for this AA
    sub_rots_dict  = AA_rot_dict[sc_AA]
    sub_rots_count = len(sub_rots_dict)

    base_sc = input_pose.residue(sc_idx)
    rot_idx_list = [x for x,y in sub_rots_dict.items()]

    ## create a list of pose replace the sc with rotamers
    sc_rots_list = []
    for rot_idx in rot_idx_list:
        chi_array = sub_rots_dict[rot_idx]
        query_rot = base_sc.clone()
        for jj in range(1, query_rot.nchi()+1):
            query_rot.set_chi(jj, chi_array[jj-1])

        aligned_rot = align_rot_to_polar(base_sc, query_rot)

        rot_pose = pyrosetta.Pose()
        rot_pose.append_residue_by_jump(aligned_rot, 1)
        # rot_pose.dump_pdb(f"{debug_dir}/rot_{rot_idx}.pdb")

        sc_rots_list.append( (rot_idx, rot_pose) )

    return sc_rots_list

def add_entry(key, chi_idx, key_count, out_dtype, out_dat):
    each_entry = np.zeros(1, dtype=out_dtype)
    each_entry["key"]     = key
    each_entry["chi_idx"] = chi_idx
    adding_memmap_array = np.memmap(out_dat, mode="r+",
                                    dtype=out_dtype,
                                    shape=(1,),
                                    offset = key_count*out_dtype.itemsize)
    adding_memmap_array[0] = each_entry[0]

def find_chi_idx(query_chi, sub_rots_dict):
    rot_idx_list = [x for x,y in sub_rots_dict.items()]
    rot_array = np.array([sub_rots_dict[x] for x in rot_idx_list])
    dist_mtx = cdist([query_chi], rot_array)
    min_idx  = np.argmin(dist_mtx)
    # print(dist_mtx[0,min_idx])
    base_hit = rot_idx_list[min_idx]

    # print(sub_rots_dict[base_hit])
    # print(query_chi)
    return base_hit, sub_rots_dict[base_hit]





def worker(input_pose, sc_rots_list, DBs_dir, input_id, jitter_n=10000, twist_n=1000):

    sc_idx = 2
    bb_idx = 5
    sc_res = input_pose.residue(sc_idx).clone()
    bb_res = input_pose.residue(bb_idx).clone()
    input_pair = pyrosetta.Pose()
    input_pair.append_residue_by_jump(sc_res, 1)
    input_pair.append_residue_by_jump(bb_res, 1)
    sc_AA = input_pair.residue(1).name1()

    ## setup hash function
    tor_resl   = 10
    cart_resl  = 1
    ori_resl   = 10
    cart_bound = 32
    hashfxn = RosettaStubTwoTorsionHash(tor_resl, cart_resl, ori_resl, cart_bound)

    ## output date type
    ## a uinit 64 key, and and uinit 16 chi idx
    out_dtype = np.dtype([("key",      "u8"),
                          ("chi_idx",  "u2")])

    ## create dat for store the raw hashtable
    out_hash_type = f"NQBB_{input_pair.residue(1).name1()}"
    out_dat = f"{DBs_dir}/raw_key_chi_idx_{out_hash_type}_{np.random.randint(1e6, 1e7)}.dat"
    while os.path.isfile(out_dat):
        out_dat = f"{DBs_dir}/raw_key_chi_idx_{out_hash_type}_{np.random.randint(1e6, 1e7)}.dat"
    out_array = np.memmap(out_dat, mode="w+", dtype=out_dtype, shape=(1,))
    key_count = 0

    scorefxn = pyrosetta.create_score_function("ref2015")
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)
    base_score = scorefxn(input_pair)

    base_sc_chain = pyrosetta.Pose(input_pose, 1, 3)
    base_bb_chain = pyrosetta.Pose(input_pose, 4, 6)

    in_npz = np.load("./PDB30_20FEB17_noGPIV_noPreP.npz")
    in_array = in_npz["pp"]
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(in_array)

    ## sample random Matrix
    M_list = [np.identity(4)]
    for ii in range(jitter_n):
        angle_sd = 60
        xyz_sd = 1
        random_M = generate_random_M(angle_sd, xyz_sd)
        M_list.append(random_M)

    ref_atoms   = ["CA", "N", "C"]
    query_atoms = ["CA", "N", "C"]

    count = 0
    # duplicate_dict = defaultdict(list)
    duplicate_dict = defaultdict(int)
    best_score = np.inf
    best_pair  = None
    for this_M in M_list:
        jittered_bb_chain = generate_new_pose_by_M(base_bb_chain, this_M)
        jittered_pair = pyrosetta.Pose()
        jittered_pair.append_residue_by_jump(base_sc_chain.residue(2).clone(),1)
        jittered_pair.append_residue_by_jump(jittered_bb_chain.residue(2).clone(), 1)

        ## for legacy reasons from the ray hash script
        ## check ray type can tell the hbond types the bb res is participating
        ray_types = check_ray_type(jittered_pair, 1, 2)

        phipsi_list = [ [jittered_bb_chain.phi(2), jittered_bb_chain.psi(2)] ]
        ## generate random phi psi values
        n_kde_sample = twist_n
        phipsi_list.extend( kde.sample(n_kde_sample) )

        ## sample randomly with changing phi psi and jump
        for this_phi, this_psi in phipsi_list:
            twisted_bb_chain = jittered_bb_chain.clone()
            twisted_bb_chain.set_phi(2,this_phi)
            twisted_bb_chain.set_psi(2,this_psi)
            twisted_bb_chain.update_residue_neighbors()

            ## the bb res of the bb_chain is 2
            ## align to parent bb chain
            twisted_bb_chain = align_to_ref_bb(jittered_bb_chain, 2, twisted_bb_chain, 2)

            twisted_pair = pyrosetta.Pose()
            twisted_pair.append_residue_by_jump(base_sc_chain.residue(2).clone(),    1)
            twisted_pair.append_residue_by_jump(twisted_bb_chain.residue(2).clone(), 1)

            ## for legacy reasons from the ray hash script
            ## check ray type can tell the hbond types the bb res is participating
            ray_types = check_ray_type(twisted_pair, 1, 2)
            if not len(ray_types) == 2: continue

            for chi_idx, sc_rot_pose in sc_rots_list:
                aligned_sc_rot = align_rot_to_polar(sc_res, sc_rot_pose.residue(1))

                out_pair = pyrosetta.Pose()
                out_pair.append_residue_by_jump(aligned_sc_rot, 1)
                out_pair.append_residue_by_jump(twisted_bb_chain.residue(2).clone(), 1)

                out_score = scorefxn(out_pair)
                if out_score > 0 or out_score > base_score + 0.5: continue

                hash_key = get_stubtor_key(hashfxn, out_pair.residue(1), out_pair.residue(2),
                                           ref_atoms,query_atoms,this_phi,this_psi)

                if hash_key in duplicate_dict: continue

                # print(hash_key)
                add_entry(hash_key, chi_idx, key_count, out_dtype, out_dat)

                if out_score < best_score:
                    best_score = out_score
                    best_pair  = out_pair.clone()
                    best_key   = hash_key
                duplicate_dict[hash_key] = 1
                key_count += 1

                if key_count%10000 == 0:
                    # best_pair = align_to_bb(best_pair, 1)
                    # best_pdb = f"local_debug/best_pair_{key_count}_{input_id}_{best_score:.2f}_{best_key}.pdb"
                    # best_pair.dump_pdb(best_pdb)
                    out_pair = align_to_bb(out_pair, 1)
                    out_pdb = f"local_debug/for_illustration_{sc_AA}_{hash_key}_{chi_idx}_{this_phi}_{this_psi}_{out_score:.2f}.pdb"
                    # out_pair.dump_pdb(out_pdb)

    return None


if __name__ == "__main__":
    pyrosetta.init("-mute all -beta")

    ## parse the input
    input_pdb = sys.argv[1]
    print(input_pdb)
    input_id = os.path.basename(input_pdb).split(".pdb")[0]
    # input_hash_type = sys.argv[2]
    # print(input_hash_type)
    input_pose = pyrosetta.pose_from_file(input_pdb)

    ## setup the directories
    pwd = os.getcwd()
    pwd = re.sub("/mnt", "", pwd)
    pwd = re.sub("/laobai/digs", "", pwd)
    scratch_dir = "/home/huabai/net_scratch/DBs_scratch"
    DBs_dir = f"{pwd}/DBs_local"
    if not os.path.isdir(DBs_dir):
        os.mkdir(DBs_dir)

    debug_dir = f"{pwd}/local_debug"
    if not os.path.exists(debug_dir):
        os.mkdir(debug_dir)

    sc_idx = 2
    bb_idx = 5
    # input_pose.dump_pdb(f"{debug_dir}/debug_input.pdb")
    sc_AA = input_pose.residue(sc_idx).name1()
    assert input_pose.pdb_info().res_haslabel(sc_idx, "SC")


    scorefxn = pyrosetta.create_score_function("ref2015")
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 1)
    scorefxn(input_pose)

    base_energies = input_pose.energies()
    ## this is one of the rare cases Rosetta use 0-based index (another example is the xyz coordinates)
    base_fa_rep   = base_energies.residue_total_energies_array()["fa_rep"][sc_idx-1]
    base_fa_atr   = base_energies.residue_total_energies_array()["fa_atr"][sc_idx-1]
    base_fa_dun   = base_energies.residue_total_energies_array()["fa_dun"][sc_idx-1]

    base_rep_atr = base_fa_rep * scorefxn.weights()[fa_rep] + \
                   base_fa_atr * scorefxn.weights()[fa_atr]

    # print(base_fa_rep)
    # print(base_fa_atr)
    # print(base_rep_atr)
    # print(base_fa_dun)

    # if base_rep_atr > 0: sys.exit()
    ## if score the base pose first, it will not change after mutation?
    ## need double check
    ## work around by creating new pose
    check_ray_type(input_pose, sc_idx, bb_idx)

    sc_res = input_pose.residue(sc_idx).clone()
    bb_res = input_pose.residue(bb_idx).clone()
    input_pair = pyrosetta.Pose()
    input_pair.append_residue_by_jump(sc_res, 1)
    input_pair.append_residue_by_jump(bb_res, 1)
    # input_pair.dump_pdb(f"{debug_dir}/debug_sc_0.pdb")

    ## checking hbond rays
    check_ray_type(input_pair, 1, 2)
    ## repeat the hbond checking
    input_hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(input_pair, calculate_derivative=False, hbond_set=input_hbset)
    for ii in [1,2]:
        hbond_type = hash_helpers.check_hbond_types(input_hbset, input_pair, ii)
        # print(hbond_type)

    ref_idx     = 1 ## use the sc_residue as reference residue
    query_idx   = 2 ## use the bb_residue as the residue to calculate key
    ref_res     = input_pair.residue(ref_idx)
    ref_atoms   = ["N", "CA", "C"]
    query_res   = input_pair.residue(query_idx)
    query_atoms = ["N", "CA", "C"]
    # query_phi   = input_pair.phi(query_idx)
    # query_psi   = input_pair.psi(query_idx)
    query_phi   = input_pose.phi(5)
    query_psi   = input_pose.psi(5)
    # print(query_phi)
    # print(query_psi)

    AA_rot_dict = load_rots_info()
    sub_rots_dict = AA_rot_dict[sc_AA]
    # base_chi = [sc_res.chi(x+1) for x in range(sc_res.nchi())]
    # base_hit, closest_chi = find_chi_idx(base_chi, sub_rots_dict)
    # print(base_hit)
    # print(closest_chi)

    ## check the base hash key
    # base_key = get_stubtor_key(hashfxn,ref_res,query_res,ref_atoms,query_atoms,query_phi,query_psi)
    # add_entry(base_key, base_hit, key_count, out_dtype, out_dat)
    # key_count += 1
    ## not add base key this time due to the chi idx issue

    ## this is the debug part, checking the input pair and changed rotamer pair
    # debug_sc = sc_res.clone()
    # for jj in range(1, debug_sc.nchi()+1):
    #     debug_sc.set_chi(jj, closest_chi[jj-1])
    #     aligned_debug = align_rot_to_polar(sc_res, debug_sc)
    # debug_key = get_stubtor_key(hashfxn,aligned_debug,query_res,ref_atoms,query_atoms,query_phi,query_psi)
    # print(base_key)
    # print(debug_key)
    # debug_pair = pyrosetta.Pose()
    # debug_pair.append_residue_by_jump(aligned_debug, 1)
    # debug_pair.append_residue_by_jump(bb_res.clone(), 1)
    # debug_pair.dump_pdb(f"{debug_dir}/debug_sc_1_{debug_key}.pdb")
    # input_pair.dump_pdb(f"{debug_dir}/debug_sc_0_{base_key}.pdb")

    sc_rots_list = generate_aligned_rots(input_pose, AA_rot_dict)
    # print( len(sc_rots_list) )

    ## sample_phipsi_and_jitters
    worker(input_pose, sc_rots_list, DBs_dir, input_id, jitter_n=10000, twist_n=100)
