#!/usr/bin/env python
# coding: utf-8

import pyrosetta
from glob import glob
import re, os,sys
import hash_helpers
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fclusterdata
from collections import defaultdict
import numpy as np
import seaborn as sns
import json
import transformations
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3
from scipy.spatial.distance import pdist
from Bio.Alphabet.IUPAC import IUPACProtein

def find_optimal_cluster_n(input_array):
    if len(input_array) < 1:
        return 0
    if len(input_array) == 1:
        return 1
    if len(input_array) == 2:
        return 2

    silhouette_scores = []
    max_n = min(10, len(input_array))
    for k in range(2, max_n): ## cluster number from 2 to max-1
        kmeans = KMeans(n_clusters = k).fit(input_array)
        labels = kmeans.labels_
        each_score = silhouette_score(input_array, labels, metric = 'euclidean')
        silhouette_scores.append(each_score)
        return np.argmax(silhouette_scores) + 2

def get_rot_arrays_for_single_residue(sfx, input_pose, res_idx, phi, psi, cluster_atoms_dict, include_current=True):

    rotamers = hash_helpers.get_rots_for_single_residue(input_pose.residue(res_idx), ori_phi=phi, ori_psi=psi, include_current=include_current)

    aligned_rots = []
    dun_array = np.zeros(len(rotamers))
    chi_array = np.zeros((len(rotamers), input_pose.residue(res_idx).nchi()))

    AA_id = input_pose.residue(res_idx).name1()
    cluster_atoms = cluster_atoms_dict[AA_id]
    cluster_array = np.zeros( ( len(rotamers), len(cluster_atoms), 3 ) )


    for ii, each_rot in enumerate(rotamers):
        rot_pose = pyrosetta.Pose()
        rot_pose.append_residue_by_jump(each_rot.clone(), 1)
        sfx(rot_pose)
        rot_energies = rot_pose.energies()
        dun_array[ii] = rot_energies.residue_total_energies_array()["fa_dun"][0]

        for jj in range(1, each_rot.nchi()+1):
            chi_array[ii,jj-1] = each_rot.chi(jj)

        aligned_rot = generate_aligned_rot(each_rot)
        aligned_rots.append(aligned_rot)

        cluster_xyz = get_res_xyz(aligned_rot, atom_names = cluster_atoms)
        cluster_array[ii] = cluster_xyz

    return aligned_rots, chi_array, dun_array, cluster_array


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

def generate_aligned_rot(raw_rot):
    # alignment_atoms_dict={}
    # alignment_atoms_dict['D'] = ["CG", "OD1", "OD2"]
    # alignment_atoms_dict['N'] = ["CG", "OD1", "ND2"]
    # alignment_atoms_dict['E'] = ["CD", "OE1", "OE2"]
    # alignment_atoms_dict['Q'] = ["CD", "OE1", "NE2"]
    # alignment_atoms_dict['R'] = ["CZ", "NH1", "NH2"]
    # alignment_atoms_dict['K'] = ["CD",  "CE",  "NZ"]
    # alignment_atoms_dict['W'] = ["CD1", "NE1", "CE2"]
    # alignment_atoms_dict['H'] = ["ND1", "CE1", "NE2"]
    # alignment_atoms_dict['Y'] = ["CG",  "CE1", "CE2"]

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

    AA_id = raw_rot.name1()
    atoms_to_align    = alignment_atoms_dict[AA_id]

    ## extract the xyz of the three atoms
    coordinates_mobile = get_res_xyz(raw_rot, atoms_to_align)

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

    input_xyz = get_res_xyz(raw_rot)
    input_xyz_addone = np.ones((input_xyz.shape[0],input_xyz.shape[1]+1))
    input_xyz_addone[:,:-1] = input_xyz

    output_xyz_addone = np.dot(M, input_xyz_addone.transpose())
    output_xyz = output_xyz_addone[:3,:].transpose()

    output_res = raw_rot.clone()
    set_res_xyz(output_res, output_xyz)
    return output_res

def rms3x3(arrayA, arrayB):
    v_AB = arrayB - arrayA
    return np.vdot(v_AB, v_AB) ** 0.5
    ## return np.sqrt( np.sum( np.square( np.linalg.norm(v_AB, axis=0) ) ) )

def rms_9d(arrayA, arrayB):
    arrayA33 = arrayA.reshape(-1,3)
    arrayB33 = arrayB.reshape(-1,3)

    v_AB = arrayB33 - arrayA33
    return np.vdot(v_AB, v_AB) ** 0.5


def get_sc_array(this_sc = "N", sc_dict=None, old_rots_dict=None):
    if sc_dict: return None
    sfx = pyrosetta.create_score_function("ref2015")
    cluster_atoms_dict={}
    cluster_atoms_dict['D'] = ["CG", "OD1", "OD2"]
    cluster_atoms_dict['N'] = ["CG", "OD1", "ND2"]
    cluster_atoms_dict['E'] = ["CD", "OE1", "OE2"]
    cluster_atoms_dict['Q'] = ["CD", "OE1", "NE2"]
    cluster_atoms_dict['R'] = ["CZ", "NH1", "NH2"]
    cluster_atoms_dict['K'] = ["CD",  "CE",  "NZ"]
    cluster_atoms_dict['W'] = ["CD1", "NE1", "CE2"]
    cluster_atoms_dict['H'] = ["ND1", "CE1", "NE2"]
    cluster_atoms_dict['Y'] = ["CG",  "CE1", "CE2"]

    print(this_sc)

    ## using the 1st trio as the base rot
    this_pdb = sc_dict[this_sc][0]
    this_pair = pyrosetta.pose_from_file(this_pdb)
    rotamers, chi_array, dun_array, cluster_array = get_rot_arrays_for_single_residue(sfx, this_pair, 1, -64, -41, cluster_atoms_dict)






    ## additional arrays and dun
    add_arrays = []
    add_dun = []
    add_cluster = []

    for this_pdb in sc_dict[this_sc][1:]:
        this_pair = pyrosetta.pose_from_file(this_pdb)
        sfx(this_pair)
        this_energies = this_pair.energies()
        add_dun.append(this_energies.residue_total_energies_array()["fa_dun"][0])

        this_rot = this_pair.residue(1).clone()

        aligned_rot = generate_aligned_rot(this_rot)

        AA_id = this_rot.name1()
        cluster_atoms = cluster_atoms_dict[AA_id]

        sub_array = np.zeros((1, this_rot.nchi()))
        for jj in range(1, this_rot.nchi()+1):
                sub_array[0,jj-1] = this_rot.chi(jj)
        add_arrays.append(sub_array)

        cluster_xyz = get_res_xyz(aligned_rot, atom_names = cluster_atoms)
        add_cluster.append( cluster_xyz.reshape((-1,len(cluster_atoms),3))  )

    ## combine all the dun_array and chi_array
    dun_array = np.hstack([dun_array] + add_dun)
    chi_array = np.vstack([chi_array] + add_arrays)
    print(cluster_array.shape)
    print(add_cluster[0].shape)
    cluster_array = np.vstack([cluster_array] + add_cluster)
    # sns.distplot(dun_array)

    ## remove the rotamers with high dun score
    max_dun = np.mean(dun_array) + np.std(dun_array)
    selected_array = cluster_array[np.where(dun_array < max_dun)]
    selected_chis = chi_array[np.where(dun_array < max_dun)]
    print(f"raw array n: {selected_array.shape[0]}")
    print(selected_array.shape)

    return selected_array, selected_chis


def allocate_AA_rots():
    aa_list = IUPACProtein.letters
    rot_limit_dict = {}
    quota_dict = {}
    for each_aa in aa_list:
        aa_pose = pyrosetta.pose_from_sequence(each_aa)
        rotamers = hash_helpers.get_rots_for_single_residue(aa_pose.residue(1), ori_phi=-64, ori_psi=-41)
        rot_limit_dict[each_aa] = len(rotamers)

    rot_total_n = sum([y for x, y in rot_limit_dict.items()])
    rot_1_count = len([x for x,y in rot_limit_dict.items() if y == 1])

    scale_factor = (np.iinfo("u2").max - rot_1_count) // (rot_total_n - rot_1_count)
    left_over = np.iinfo("u2").max - rot_1_count - scale_factor * (rot_total_n - rot_1_count)
    print(left_over)
    chosen_aa = "DEFHKNQRSTWY"
    add_slots = left_over // len(chosen_aa) + 1

    idx_start = 0
    for aa_id, base_count in rot_limit_dict.items():
        if base_count == 1:
            quota_n = base_count
        elif aa_id in chosen_aa:
            quota_n = base_count * scale_factor + add_slots
        else:
            quota_n = base_count * scale_factor

        slot_start = idx_start
        slot_end   = slot_start + quota_n
        if slot_end > np.iinfo("u2").max:
            slot_end = np.iinfo("u2").max
        idx_start = slot_end

        quota_dict[aa_id] = [slot_start, slot_end]

    return quota_dict



def generate_cluster_atoms():
    cluster_atoms_dict={}
    cluster_atoms_dict['D'] = ["CG", "OD1", "OD2"]
    cluster_atoms_dict['N'] = ["CG", "OD1", "ND2"]
    cluster_atoms_dict['E'] = ["CD", "OE1", "OE2"]
    cluster_atoms_dict['Q'] = ["CD", "OE1", "NE2"]
    cluster_atoms_dict['R'] = ["CZ", "NH1", "NH2"]
    cluster_atoms_dict['K'] = ["CD",  "CE",  "NZ"]
    cluster_atoms_dict['W'] = ["CD1", "NE1", "CE2"]
    cluster_atoms_dict['H'] = ["ND1", "CE1", "NE2"]
    cluster_atoms_dict['Y'] = ["CG",  "CE1", "CE2"]
    return cluster_atoms_dict

def generate_rots_with_packer(sc_name1, sc_idx, ref_pair, include_current=True):
    cluster_atoms_dict = generate_cluster_atoms()
    sfx = pyrosetta.create_score_function("ref2015")

    assert ref_pair.residue(sc_idx).name1() == sc_name1
    sc_phi = ref_pair.phi(sc_idx)
    sc_psi = ref_pair.psi(sc_idx)

    rotamers, chi_array, dun_array, xyz_array = get_rot_arrays_for_single_residue(sfx, ref_pair, sc_idx, sc_phi, sc_psi, cluster_atoms_dict, include_current=include_current)

    return rotamers, chi_array, dun_array, xyz_array

def collect_additional_rots(sc_name1, sc_dict, sc_idx):
    cluster_atoms_dict = generate_cluster_atoms()
    sfx = pyrosetta.create_score_function("ref2015")

    ## additional arrays and dun
    add_rots = []
    add_chi  = []
    add_dun  = []
    add_xyz  = []

    for this_pdb in sc_dict[sc_name1][1:]:
        this_pair = pyrosetta.pose_from_file(this_pdb)
        sfx(this_pair)
        this_energies = this_pair.energies()
        add_dun.append(this_energies.residue_total_energies_array()["fa_dun"][sc_idx-1])

        this_rot = this_pair.residue(sc_idx).clone()
        aligned_rot = generate_aligned_rot(this_rot)

        AA_id = this_rot.name1()
        cluster_atoms = cluster_atoms_dict[AA_id]

        sub_array = np.zeros( this_rot.nchi() )
        for jj in range(1, this_rot.nchi()+1):
                sub_array[jj-1] = this_rot.chi(jj)
        add_chi.append(sub_array)

        cluster_xyz = get_res_xyz(aligned_rot, atom_names = cluster_atoms)
        add_xyz.append( cluster_xyz.reshape((len(cluster_atoms),3)) )

        add_rots.append(aligned_rot)
    return add_rots, add_chi, add_dun, add_xyz



def get_rots_of_AA(aa_id):
    aa_pose = pyrosetta.pose_from_sequence(aa_id)
    rotamers,chi_array,dun_array,xyz_array= generate_rots_with_packer(aa_id, 1, aa_pose, include_current=False)

    return rotamers, chi_array, xyz_array

def prepare_old_chi_array(this_sc, old_rots_dict):
    aa_pose = pyrosetta.pose_from_sequence(this_sc)
    chi_n = aa_pose.residue(1).nchi()
    cluster_atoms_dict = generate_cluster_atoms()
    cluster_atoms = cluster_atoms_dict[this_sc]

    chi_array_dt = np.dtype([ ("chi_idx", "u2"),
                              ("chi_set", 'f4', (chi_n,)),
                              ("xyz_set", 'f4', (3,3)) ])

    old_chis_dict = old_rots_dict[this_sc]

    old_chi_array = np.zeros( len(old_chis_dict), dtype=chi_array_dt)
    for ii, (chi_idx, chi_set) in enumerate( old_chis_dict.items() ):
        old_chi_array[ii]["chi_idx"] = chi_idx
        old_chi_array[ii]["chi_set"] = chi_set

        this_res = aa_pose.residue(1).clone()
        for jj in range(1, this_res.nchi()+1):
            this_res.set_chi(jj, chi_set[jj-1])

        aligned_rot = generate_aligned_rot(this_res)
        aligned_xyz = get_res_xyz(aligned_rot, atom_names=cluster_atoms)
        old_chi_array[ii]["xyz_set"] = aligned_xyz

    return old_chi_array

def cluster_combined_array(xyz_array, chi_array):
    flatten_array = xyz_array.reshape(xyz_array.shape[0],-1)
    t_cutoff = 0.8
    ## cluster the rest of the rotamers by distance
    clusters_idx = fclusterdata(flatten_array, t=t_cutoff, criterion="distance", method="ward")
    unique_clusters = list(set(clusters_idx))
    cluster_n = len(unique_clusters)
    # print(f"Final cluster n: {cluster_n}")

    chi_n = chi_array.shape[-1]
    chi_array_dt = np.dtype([ ("chi_idx", "u2"),
                              ("chi_set", 'f4', (chi_n,)),
                              ("xyz_set", 'f4', (3,3)) ])
    new_chi_array = np.zeros( cluster_n, dtype=chi_array_dt)

    for ii, cluster_i in enumerate(unique_clusters):
        array_pos = np.where(clusters_idx == cluster_i)[0]

        sub_chi_array = chi_array[array_pos]
        sub_xyz_array = xyz_array[array_pos]

        center_chi_array = np.average(sub_chi_array, axis=0).reshape(1,-1)
        dist_info  = cdist(sub_chi_array, center_chi_array, metric="euclidean")
        center_idx = np.argmin(dist_info)

        new_chi_array[ii]["chi_idx"] = cluster_i
        new_chi_array[ii]["chi_set"] = sub_chi_array[center_idx]
        new_chi_array[ii]["xyz_set"] = sub_xyz_array[center_idx]

    return new_chi_array

if __name__ == "__main__":
    pyrosetta.init("-mute all")

    if not os.path.isfile("rot_idx_allocation.json"):
        allocation_dict = allocate_AA_rots()
        with open("rot_idx_allocation.json", "w") as f:
            json.dump(allocation_dict, f)
    else:
        with open("rot_idx_allocation.json", "r") as f:
            allocation_dict = json.load(f)
    print(allocation_dict)


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

    ## collect all the pdbs for generating new rots
    input_dict = defaultdict(list)
    for a,b,c in os.walk("./pairs_seeds"):
        base_a = os.path.basename(a)
        for f in c:
            assert f.endswith(".pdb")
            input_dict[base_a].append(f"{a}/{f}")

    sc_list = list (set( [x[0] for x in input_dict.keys()] ))
    sc_dict = {}
    for each_sc in sc_list:
        input_pdbs = []
        for hash_type, pdbs_list in input_dict.items():
            if hash_type[0] == each_sc:
                input_pdbs.extend(pdbs_list)
        sc_dict[each_sc] = input_pdbs

    for this_sc in sc_list:
        ## for each sc id
        ## there are 3 groups of rots:
        ## group1: using Rosetta to generate rotamers
        ## group2: collect the native rotamers from pdbs
        ## group3: loaded previous rotamers
        print(this_sc)

        ## group1: Rosetta generated_rots:
        base_rots, base_chi_array, base_xyz_array = get_rots_of_AA(this_sc)

        ## group2: get from the collect pairs_seeds
        sc_idx   = 2 ## if using pair of tripeptide, the sc_idx is 2
        np_file = f"xyz_array_and_chi_array_for_{this_sc}.npy"
        if not os.path.isfile(np_file):
            add_rots, add_chi_array, add_dun_array, add_xyz_array = collect_additional_rots(this_sc, sc_dict, sc_idx)
            with open(np_file, "wb") as f:
                np.save(f, add_xyz_array)
                np.save(f, add_chi_array)
                np.save(f, add_dun_array)
        else:
            with open(np_file, "rb") as f:
                add_xyz_array = np.load(f)
                add_chi_array = np.load(f)
                add_dun_array = np.load(f)

        ## filter the array by dun score
        max_dun = np.mean(add_dun_array) + np.std(add_dun_array)
        add_xyz_array = add_xyz_array[np.where(add_dun_array < max_dun)]
        add_chi_array = add_chi_array[np.where(add_dun_array < max_dun)]

        combined_xyz = np.vstack([base_xyz_array, add_xyz_array])
        combined_chi = np.vstack([base_chi_array, add_chi_array])

        new_chi_array = cluster_combined_array(combined_xyz, combined_chi)

        ## group3: load the previous rotamers from json
        old_chi_array = prepare_old_chi_array(this_sc, old_rots_dict)

        if len(old_chi_array) > 0:
            ## compare the new_chi_array with the old_chi_array
            ## and append the new_chi into the old_chi
            rot_idx_start = old_chi_array["chi_idx"].max() + 1
            rot_idx_max   = allocation_dict[this_sc][-1] - 1

            new_rot_idx = rot_idx_start
            AB_mtx = cdist(new_chi_array["chi_set"], old_chi_array["chi_set"])

            out_array = np.copy(old_chi_array)
            for A_idx in range(AB_mtx.shape[0]):
                A_chi_set = new_chi_array[A_idx]["chi_set"]
                dist_to_B = AB_mtx[A_idx]
                if dist_to_B.min() > 2:
                    if new_rot_idx <= rot_idx_max:
                        out_array = np.hstack([out_array, new_chi_array[A_idx] ])
                        out_array[-1]["chi_idx"] = new_rot_idx
                        new_rot_idx += 1

        else:
            ## compare the new_chi_array with the old_chi_array
            ## and append the new_chi into the old_chi
            rot_idx_start = allocation_dict[this_sc][0]
            rot_idx_max   = allocation_dict[this_sc][-1] - 1
            new_rot_idx = rot_idx_start

            out_array = np.copy(new_chi_array)
            for A_idx in range(new_chi_array.shape[0]):
                out_array[A_idx]["chi_idx"] = new_rot_idx
                new_rot_idx += 1

        sub_dict = {}
        for line_idx in range(out_array.shape[0]):
            chi_idx = out_array[line_idx]["chi_idx"]
            chi_set = out_array[line_idx]["chi_set"]
            sub_dict[int(chi_idx)] = chi_set.tolist()


        old_rots_dict[this_sc] = sub_dict

    with open("rotamers.json", "w") as f:
        json.dump(old_rots_dict, f)





        # for cluster_i in list(set(clusters_idx)):
        #     # print(cluster_i)
        #     array_pos = np.where(clusters_idx == cluster_i)[0]
        #     out_pose = pyrosetta.Pose()
        #     for each_pos in array_pos:
        #         this_res = base_rots[0].clone()
        #         for jj in range(1, this_res.nchi()+1):
        #             this_res.set_chi(jj, combined_chi[each_pos][jj-1])
        #         out_pose.append_residue_by_jump(this_res, 1)
        #     out_pose.dump_pdb(f"temp/cluster_check_{cluster_i}_{len(array_pos)}.pdb")
