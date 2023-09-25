# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from datasets import lane_segment, load_xml
import copy
from os.path import join as pjoin

from typing import List, Optional, Tuple



class CarlaDataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 radius: float = 75,
                 local_radius: float = 30) -> None:
        self._split = split
        self._radius = radius
        self._local_radius = local_radius
        self._directory = "scene_mining_cav/mpr0/"
        # if split == 'train':
        #     self._directory = 'train'
        # elif split == 'val':
        #     self._directory = 'val'
        # elif split == 'test':
        #     self._directory = 'test'
        # else:
        #     raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = [f for f in os.listdir(self.raw_dir) if 'scene' in f]
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names if 'scene' in f]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(CarlaDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, self._split)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'scene_mining_hivt/processed', self._split)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        self.get_map_polygon_bbox()
        for raw_path in tqdm(self.raw_paths):
            kwargs = self.process_carla(self._split, raw_path, self._radius, self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])

    def get_map_polygon_bbox(self):
        rel_path = "Town03.osm"
        roads = load_xml.load_lane_segments_from_xml(pjoin(self.root, rel_path))
        polygon_bboxes, lane_starts, lane_ends = load_xml.build_polygon_bboxes(roads)
        self.roads = roads
        self.polygon_bboxes = polygon_bboxes
        self.lane_starts = lane_starts
        self.lane_ends = lane_ends

    def process_carla(self,
                      split: str,
                      raw_path: str,
                      radius: float,
                      local_radius: float) -> Dict:
        df = pd.read_csv(raw_path)

        # filter out actors that are unseen during the historical time steps
        timestamps = list(np.sort(df['frame'].unique()))
        historical_timestamps = timestamps[: 50]
        historical_df = df[df['frame'].isin(historical_timestamps)]
        actor_ids = list(historical_df['vid'].unique())
        df = df[df['vid'].isin(actor_ids)]
        num_nodes = len(actor_ids)

        objs = df.groupby(['vid', 'obj_type_mpr_02', 'obj_type_mpr_04', 'obj_type_mpr_06', 'obj_type_mpr_08', 'in_av_range']).groups
        keys = list(objs.keys())

        obj_type_02 = [x[1] for x in keys]
        obj_type_04 = [x[2] for x in keys]
        obj_type_06 = [x[3] for x in keys]
        obj_type_08 = [x[4] for x in keys]
        in_av_range = [x[5] for x in keys]

        cav_df = df[df['obj_type_mpr_02'] == 'cav'].iloc
        cav_index = actor_ids.index(cav_df[0]['vid'])

        # make the scene centered at CAV
        origin = torch.tensor([cav_df[49]['position_x'], cav_df[49]['position_y']], dtype=torch.float)
        cav_heading_vector = origin - torch.tensor([cav_df[48]['position_x'], cav_df[48]['position_y']], dtype=torch.float)
        theta = torch.atan2(cav_heading_vector[1], cav_heading_vector[0])
        rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]])

        # initialization
        x = torch.zeros(num_nodes, 100, 2, dtype=torch.float)
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        padding_mask = torch.ones(num_nodes, 100, dtype=torch.bool)
        bos_mask = torch.zeros(num_nodes, 50, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

        for actor_id, actor_df in df.groupby('vid'):
            node_idx = actor_ids.index(actor_id)
            node_steps = [timestamps.index(timestamp) for timestamp in actor_df['frame']]
            padding_mask[node_idx, node_steps] = False
            if padding_mask[node_idx, 49]:  # make no predictions for actors that are unseen at current timestep
                padding_mask[node_idx, 50:] = True
            xy = torch.from_numpy(np.stack([actor_df['position_x'].values, actor_df['position_y'].values], axis=-1)).float() #[100,2]
            x[node_idx, node_steps] = torch.matmul(rotate_mat, (xy - origin.reshape(-1, 2)).T).T
            node_historical_steps = list(filter(lambda node_step: node_step < 50, node_steps))
            if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
                heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
            else:  # make no predictions for the actor if the number of valid time steps is less than 2
                padding_mask[node_idx, 50:] = True

        # bos_mask is True if time step t is valid and time step t-1 is invalid
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1: 50] = padding_mask[:, : 49] & ~padding_mask[:, 1: 50]

        #positions are transformed absolute x, y coordinates
        positions = x.clone()

        x[:, 50:] = torch.where((padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
                                torch.zeros(num_nodes, 50, 2),
                                x[:, 50:] - x[:, 49].unsqueeze(-2))
        x[:, 1: 50] = torch.where((padding_mask[:, : 49] | padding_mask[:, 1: 50]).unsqueeze(-1),
                                    torch.zeros(num_nodes, 49, 2),
                                    x[:, 1: 50] - x[:, : 49])
        x[:, 0] = torch.zeros(num_nodes, 2)

        # get lane features at the current time step
        lane_pos, lane_vectors, lane_idcs, lane_actor_index, lane_actor_attr = \
                self.get_lane_feats(origin, rotate_mat, num_nodes, positions, radius, local_radius)

        y = None if split == 'test' else x[:, 50:]
        seq_id = os.path.splitext(os.path.basename(raw_path))[0]

        return {
            'x': x[:, : 50],  # [N, 50, 2]
            'positions': positions,  # [N, 50, 2]
            'edge_index': edge_index,  # [2, N x N - 1]
            'y': y,  # [N, 50, 2]
            'num_nodes': num_nodes,
            'padding_mask': padding_mask,  # [N, 100]
            'bos_mask': bos_mask,  # [N, 50]
            'rotate_angles': rotate_angles,  # [N]
            'lane_vectors': lane_vectors,  # [L, 2]
            'lane_positions': lane_pos, # [L, 2]
            'lane_idcs': lane_idcs, #[L]
            'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
            'lane_actor_attr': lane_actor_attr,  # [E_{A-L}, 2]
            'seq_id': seq_id, #str
            'cav_index': cav_index,
            'origin': origin.unsqueeze(0),
            'theta': theta,
        }

    def get_lane_feats(self, origin, rotate_mat, num_nodes, positions, radius=75, local_radius=30):

        road_ids = load_xml.get_road_ids_in_xy_bbox(self.polygon_bboxes, self.lane_starts, self.lane_ends, self.roads, origin[0], origin[1], radius)
        road_ids = copy.deepcopy(road_ids)

        lanes=dict()
        for road_id in road_ids:
            road = self.roads[road_id]
            ctr_line = torch.from_numpy(np.stack(((self.roads[road_id].l_bound[:,0]+self.roads[road_id].r_bound[:,0])/2, 
                            (self.roads[road_id].l_bound[:,1]+self.roads[road_id].r_bound[:,1])/2),axis=-1))
            ctr_line = torch.matmul(rotate_mat, (ctr_line - origin.reshape(-1, 2)).T.float()).T

            x, y = ctr_line[:,0], ctr_line[:,1]
            # if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
            #     continue
            # else:
            """getting polygons requires original centerline"""
            polygon, _, _ = load_xml.build_polygon_bboxes({road_id: self.roads[road_id]})
            polygon_x = torch.from_numpy(np.array([polygon[:,0],polygon[:,0],polygon[:,2],polygon[:,2],polygon[:,0]]))
            polygon_y = torch.from_numpy(np.array([polygon[:,1],polygon[:,3],polygon[:,3],polygon[:,1],polygon[:,1]]))
            polygon_reshape = torch.cat([polygon_x,polygon_y],dim=-1) #shape(5,2)

            road.centerline = ctr_line
            road.polygon = torch.matmul(rotate_mat, (polygon_reshape.float() - origin.reshape(-1, 2)).T).T
            lanes[road_id] = road

        lane_ids = list(lanes.keys())
        lane_pos, lane_vectors = [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline

            # lane_ctrs.append(torch.from_numpy(np.asarray((ctrln[:-1]+ctrln[1:])/2.0, np.float32)))#lane center point
            # lane_vectors.append(torch.from_numpy(np.asarray(ctrln[1:]-ctrln[:-1], np.float32))) #length between waypoints
            lane_pos.append(ctrln[:-1])#lane starting point
            lane_vectors.append(ctrln[1:]-ctrln[:-1])#length between waypoints

        lane_idcs = []
        count = 0
        for i, position in enumerate(lane_pos):
            lane_idcs.append(i*torch.ones(len(position)))
            count += len(position)

        lane_idcs = torch.cat(lane_idcs, dim=0)
        lane_pos = torch.cat(lane_pos, dim=0)
        lane_vectors = torch.cat(lane_vectors, dim=0)

        lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), \
                                                            torch.arange(num_nodes)))).t().contiguous()
        lane_actor_attr = \
        lane_pos[lane_actor_index[0]] - positions[:,49,:][lane_actor_index[1]]
        mask = torch.norm(lane_actor_attr, p=2, dim=-1) < local_radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_attr = lane_actor_attr[mask]


        return lane_pos, lane_vectors, lane_idcs, lane_actor_index, lane_actor_attr

class TemporalData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_attr: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 cav_index: Optional[int] = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_vectors=lane_vectors, lane_actor_index=lane_actor_index, 
                                           cav_index=cav_index,lane_actor_attr=lane_actor_attr,
                                           seq_id=seq_id, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)
