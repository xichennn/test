# %%
import sys
import os
import os.path as osp
import re
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import gc
from copy import deepcopy, copy

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

sys.path.append('/Users/xichen/Documents/paper2-traj-pred/carla-data')
# sys.path.append("core/dataloader")
# %%
def get_noisy_feats(feats, var=0.1):
    """
    get noisy feats of neighbors
    feats: numpy ndarray of shape(a1, a2, a3)
    for ["feats"] feature: a3=3
    for ["gt_preds"] feature: a3=2
    """
    random.seed(30)
    a1, a2, a3 = feats.shape
    noise = np.random.normal(0, var, (a1, a2, 2))
    noise_ = np.concatenate((noise, np.zeros((a1, a2, 1))), axis=2)

    if a3 == 3:
        noisy_feats = feats+noise_
    elif a3 == 2:
        noisy_feats = feats+noise
    return noisy_feats

def get_shifted_feats(feats, offset=1):
    """
    get shifted feats of neighbors
    feats: numpy ndarray of shape(a1, a2, a3)
    for ["feats"] feature: a3=3
    for ["gt_preds"] feature: a3=2
    offset: number of frames
    """
    a1, a2, a3 = feats.shape
    #shift left for num of frames
    first_obss = feats[:,0:offset,:2] #(a1,2) cv
    offset_frames = feats[:,[offset],:2] - first_obss #(a1, 2)
    first_obs_shift = first_obss - offset_frames
    #shift neighbors to the last frame (10Hz, 0.1s)
    # shifted_traj = np.concatenate((np.expand_dims(first_obs_shift, 1), feats[:,:-offset,:2]), axis=1) #(a1,a2,2)
    shifted_traj = np.concatenate((first_obs_shift, feats[:,:-offset,:2]), axis=1) #(a1,a2,2)

    if a3 == 3:
        shifted_feats = np.concatenate((shifted_traj, np.expand_dims(feats[:,:,2], 2)), axis=2)
    elif a3 == 2:
        shifted_feats = shifted_traj
    return shifted_feats

def get_fc_edge_index(node_indices):
    """
    params:
    node_indices: np.array([indices]), the indices of nodes connecting of each other;
    return:
    xy: a tensor(2, edges), indicing edge_index
    """

    xx, yy = np.meshgrid(node_indices, node_indices)
    xy = np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)
    return xy

def get_traj_edge_index(node_indices):
    """
    generate the polyline graph for traj, each node are only directionally connected with the
    nodes in its future
    params:
    node_indices: np.array([indices]), the indices of nodes connecting of each other;
    return:
    a tensor(2, edges), indicing edge_index
    """
    edge_index = np.empty((2, 0))
    for i in range(len(node_indices)):
        xx, yy = np.meshgrid(node_indices[i], node_indices[i:])
        edge_index = np.hstack([edge_index, 
                                np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)])
        return edge_index

class GraphData(Data):
    """
    override key 'cluster' indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0
        
# dataset loader which loads data into memory
class CarlaInMem(InMemoryDataset):
    def __init__(self, root, idx, transform=None, pre_transform=None):
        super(CarlaInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[idx])
        gc.collect()
    
    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "scene" in file and file.endswith(".pkl")]
    
    @property
    def processed_file_names(self):
        return ['data_all.pt', 'data_av.pt', 'data_cav_02.pt', 'data_cav_04.pt', 'data_cav_06.pt', 'data_cav_08.pt']
    
    def download(self):
        pass

    def process(self):
        """ transform the raw data and store in GraphData"""
        idx = 0
        while idx<6:
            # loading the raw data
            traj_lens = []
            valid_lens = []
            candidate_lens = []
            for raw_path in tqdm(self.raw_paths, desc="Loading Raw Data..."):
                # raw_data = pd.read_pickle(raw_path)
                raw_data = self._get_raw_data(raw_path, idx)
            
                #statistics
                traj_num = raw_data["feats"].values[0].shape[0] #shape[num_vehs, hist_steps, num_feats]
                lane_num = raw_data['graph'].values[0]['lane_idcs'].max() + 1
                candidate_num = raw_data['tar_candts'].values[0].shape[0]
                # traj_num, lane_num, candidate_num = self._get_statistics(raw_data)
                traj_lens.append(traj_num)
                valid_lens.append(traj_num + lane_num)
                candidate_lens.append(candidate_num)
            num_valid_len_max = np.max(valid_lens)
            num_candidate_max = np.max(candidate_lens)
            print("\n[Argoverse]: The maximum of valid length is {}".format(num_valid_len_max))
            print("[Argoverse]: The maximum of no. of candidates is {}.".format(num_candidate_max))
        
            # pad vectors to the largest polyline id and extend cluster, save the Data to disk
            data_list = []
            for ind, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData...")):
                # raw_data = pd.read_pickle(raw_path)
                raw_data = self._get_raw_data(raw_path, idx)

                # input data
                x, cluster, edge_index, identifier = self._get_x(raw_data)
                y = self._get_y(raw_data)
                graph_input = GraphData(
                    x=torch.from_numpy(x).float(),
                    y=torch.from_numpy(y).float(),
                    cluster=torch.from_numpy(cluster).short(),
                    edge_index=torch.from_numpy(edge_index).long(),
                    identifier=torch.from_numpy(identifier).float(),    # the identify embedding of global graph completion

                    traj_len=torch.tensor([traj_lens[ind]]).int(),            # number of traj polyline
                    valid_len=torch.tensor([valid_lens[ind]]).int(),          # number of valid polyline
                    time_step_len=torch.tensor([num_valid_len_max]).int(),    # the maximum of no. of polyline

                    candidate_len_max=torch.tensor([num_candidate_max]).int(),
                    candidate_mask=[],
                    candidate=torch.from_numpy(raw_data['tar_candts'].values[0]).float(),
                    candidate_gt=torch.from_numpy(raw_data['gt_candts'].values[0]).bool(),
                    offset_gt=torch.from_numpy(raw_data['gt_tar_offset'].values[0]).float(),
                    target_gt=torch.from_numpy(raw_data['gt_preds'].values[0][0][-1, :]).float(),

                    orig=torch.from_numpy(raw_data['cav_orig'].values[0]).float().unsqueeze(0),
                    rot=torch.from_numpy(raw_data['rot'].values[0]).float().unsqueeze(0),
                    seq_id=torch.tensor([int(raw_data['seq_id'])]).int()
                )
                data_list.append(graph_input)

            # Store the processed data
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[idx])
            idx += 1
    

    def get(self, idx):
        """
        return datapoint in each dataset
        type in [0,1,2]
        """
        # self.get(type)
        data = super(CarlaInMem, self).get(idx).clone()

        feature_len = data.x.shape[1]
        index_to_pad = data.time_step_len[0].item()
        valid_len = data.valid_len[0].item()

        # pad feature with zero nodes
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len, index_to_pad, dtype=data.cluster.dtype)]).long()
        data.identifier = torch.cat([data.identifier, torch.zeros((index_to_pad - valid_len, 2), dtype=data.identifier.dtype)])

        # pad candidate and candidate_gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate[:, :2], torch.zeros((num_cand_max - len(data.candidate), 2))])
        data.candidate_gt = torch.cat([data.candidate_gt,
                                       torch.zeros((num_cand_max - len(data.candidate_gt), 1), dtype=data.candidate_gt.dtype)])

        assert data.cluster.shape[0] == data.x.shape[0], "[ERROR]: Loader error!"

        return data
    
    @staticmethod
    def _get_raw_data(raw_path, idx):
        """
        reconstruct the features of selected neighhbors in raw_data
        #list: trajs, steps, obj_type, in_av_range, ref_ctr_lines, 
        #numpy.ndarray: cav_orig, rot, feats, has_obss, has_preds, gt_preds, tar_candts, gt_candts, gt_tar_offset, 
        #numpy.float32: theta
        #numpy.int64: "ref_cetr_idx", "seq_id"
        #dict: graph
        add category to df columns(0: cav, 1:av, 2:cv)
        """
        if idx == 0: #get all neighbors in cv range
            raw_data = pd.read_pickle(raw_path)
            raw_data["category"] = pd.Series([[0]+[1]*(len(raw_data["in_av_range"].values[0])-1)])

        elif idx == 1: #get all neighbors only in av range
            raw_data0 = pd.read_pickle(raw_path)
            raw_data1 = {}
            obj_len = len(raw_data0["in_av_range"].values[0])
            for key in raw_data0.keys():
                if key in ["trajs", "steps", "obj_type_mpr_02", "obj_type_mpr_04", "obj_type_mpr_06", "obj_type_mpr_08", "in_av_range"]:
                    raw_data1[key] = pd.Series([[raw_data0[key].values[0][i] for i in range(obj_len) if raw_data0["in_av_range"].values[0][i]==True]])
                elif key in ["has_obss", "has_preds"]:
                    raw_data1[key] = pd.Series([np.asarray([raw_data0[key].values[0][i] for i in range(obj_len) if raw_data0["in_av_range"].values[0][i]==True])])
                elif key in ["feats", "gt_preds"]:
                    noisy_feats_neighbors = get_noisy_feats(raw_data0[key].values[0][1:,:,:], var=0.1)
                    merge_with_cav = np.concatenate((np.expand_dims(raw_data0[key].values[0][0,:,:], 0), noisy_feats_neighbors), axis=0)
                    raw_data1[key] = pd.Series([np.asarray([merge_with_cav[i] for i in range(obj_len) if raw_data0["in_av_range"].values[0][i]==True])])
                else:
                    raw_data1[key] = raw_data0[key]
            raw_data1["category"] = pd.Series([[0]+[1]*(len(raw_data1["in_av_range"].values[0])-1)])
            raw_data = pd.DataFrame(raw_data1)
        
        elif idx in [2, 3, 4, 5]: #get all neighbors in cav range, mpr=0.2
            mpr_keys = ["obj_type_mpr_02", "obj_type_mpr_04", "obj_type_mpr_06", "obj_type_mpr_08"]
            raw_data0 = pd.read_pickle(raw_path)
            raw_data1 = {}
            obj_len = len(raw_data0["in_av_range"].values[0])

            idcs = [] #indices for vids that are both av and cv
            category = [] #0:cav, 1:av,2:cv,3:av and cv
            #find vid that are both cv and av
            for i in range(obj_len):
                if raw_data0[mpr_keys[idx-2]].values[0][i] == "cav":
                    category.append(0)
                elif raw_data0["in_av_range"].values[0][i] == True:
                    category.append(1)
                    if raw_data0[mpr_keys[idx-2]].values[0][i] == "cv":
                        idcs.append(i)
                elif raw_data0["in_av_range"].values[0][i] == False and raw_data0[mpr_keys[idx-2]].values[0][i] == "cv":
                    category.append(2)
                
            for key in raw_data0.keys():
                if key in ["trajs", "steps", "obj_type_mpr_02", "obj_type_mpr_04", "obj_type_mpr_06", "obj_type_mpr_08", "in_av_range"]:
                    orig_features = [raw_data0[key].values[0][i] for i in range(obj_len) 
                                                if raw_data0["in_av_range"].values[0][i]==True or
                                                    (raw_data0["in_av_range"].values[0][i]==False and 
                                                    raw_data0[mpr_keys[idx-2]].values[0][i]=="cv")]
                    duplicate_features = [raw_data0[key].values[0][i] for i in idcs] #cv error
                    raw_data1[key] = pd.Series([orig_features + duplicate_features])
                elif key in ["has_obss", "has_preds"]:
                    orig_features = [raw_data0[key].values[0][i] for i in range(obj_len) 
                                                    if raw_data0["in_av_range"].values[0][i]==True or
                                                    (raw_data0["in_av_range"].values[0][i]==False and 
                                                    raw_data0[mpr_keys[idx-2]].values[0][i]=="cv")]
                    duplicate_features = [raw_data0[key].values[0][i] for i in idcs] #cv error
                    raw_data1[key] = pd.Series([np.asarray(orig_features + duplicate_features)])
                elif key in ["feats", "gt_preds"]: #inject different errors according to category
                    orig_features = [raw_data0[key].values[0][i] for i in range(obj_len) 
                                                    if raw_data0["in_av_range"].values[0][i]==True or
                                                    (raw_data0["in_av_range"].values[0][i]==False and 
                                                    raw_data0[mpr_keys[idx-2]].values[0][i]=="cv")]
                    orig_features_add_errors = []
                    for j in range(len(category)):
                        if category[j]==0:
                            orig_features_add_errors.append(orig_features[j]) 
                        elif category[j]==1:
                            orig_features_add_errors.append(np.squeeze(get_noisy_feats(np.expand_dims(orig_features[j],0), var=0.1),0))
                        elif category[j]==2:
                            orig_features_add_errors.append(np.squeeze(get_shifted_feats(np.expand_dims(orig_features[j],0), offset=1),0))
                    duplicate_features_add_errors = [np.squeeze(get_shifted_feats(np.expand_dims(raw_data0[key].values[0][j],0),offset=1),0) for j in idcs] #cv error
                    raw_data1[key] = pd.Series([np.asarray(orig_features_add_errors+duplicate_features_add_errors)])
            
                else:
                    raw_data1[key] = raw_data0[key]
            raw_data1["category"] = pd.Series([category + [2]*len(idcs)])
            raw_data = pd.DataFrame(raw_data1)

        return raw_data

    @staticmethod
    def _get_x(data_seq):
        """
        feat: [xs, ys, vec_x, vec_y, step(timestamp), polyline_id];
        xs, ys: the control point of the vector, for trajectory, it's start point, for lane segment, it's the center point;
        vec_x, vec_y: the length of the vector in x, y coordinates;
        step: indicating the step of the trajectory, for the lane node, it's always 0;
        polyline_id: the polyline id of this node belonging to;
        """

        feats = np.empty((0, 6))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))
        
        #get traj features
        traj_feats = data_seq['feats'].values[0]
        traj_has_obss = data_seq['has_obss'].values[0]
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0
        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2] #vector start 
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2] #vector length
            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], polyline_id])])
            traj_cnt += 1

        #get lane features
        graph = data_seq['graph'].values[0]
        ctrs = graph['ctrs']
        vec = graph['feats']
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt
        steps = np.zeros((len(lane_idcs), 1))
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, lane_idcs])])

        #get the cluster and construct subgraph edge_index
        cluster = copy(feats[:, -1].astype(np.int64)) #polyline_id & lane_id
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])
            if len(indices) <= 1:
                continue                # skip if only 1 node
            if cluster_idc < traj_cnt:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
            else:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
        return feats, cluster, edge_index, identifier
    
    @staticmethod
    def _get_y(data_seq):
        traj_obs = data_seq['feats'].values[0][0]
        traj_fut = data_seq['gt_preds'].values[0][0]
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        return offset_fut.reshape(-1).astype(np.float32)
    
if __name__ == "__main__":
    #for folder in os.listdir("../../scene_mining_intermediate")
    INTERMEDIATE_DATA_DIR = "test"

    dataset_input_path = os.path.join(os.path.dirname(__file__), INTERMEDIATE_DATA_DIR)
    dataset = CarlaInMem(dataset_input_path, 0).shuffle()
    batch_iter = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True, pin_memory=False)
    for k in range(1):
        for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
            pass


