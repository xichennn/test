import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
import torch 

COLOR_DICT = {"CAV": "#d33e4c", "CV": "g", "NCV": "darkorange"}

def show_doubled_lane(road):
    """
    params: road object, with attributes:
            l_bound: The coordinates of the lane segment's left bound.
            r_bound: The coordinates of the lane segment's right bound.
    """
    xl, yl = road.l_bound[:, 0], road.l_bound[:, 1]
    xr, yr = road.r_bound[:, 0], road.r_bound[:, 1]

    plt.plot(xl, yl, color="0.7")
    plt.plot(xr, yr, color="0.7")

def show_centerline(road):
    """
    params: road object, with attributes:
            l_bound: The coordinates of the lane segment's left bound.
            r_bound: The coordinates of the lane segment's right bound.
    """
    xl, yl = road.l_bound[:, 0], road.l_bound[:, 1]
    xr, yr = road.r_bound[:, 0], road.r_bound[:, 1]
    centerline = np.stack(((xl+xr)/2, (yl+yr)/2),axis=-1)

    plt.plot(centerline[:,0], centerline[:,1], "--", color="0.7")

def show_traj(traj, type_):
    """
    args: ndarray in shape of (n, 2)
    """
    plt.plot(traj[:, 0], traj[:, 1], color=COLOR_DICT[type_])

# def reconstruct_polyline(features, traj_mask, lane_mask, add_len):
#     traj_ls, lane_ls = [], []
#     for id_, mask in traj_mask.items():
#         data = features[mask[0]: mask[1]]
#         traj = np.vstack((data[:,0:2], data[-1, 2:4]))
#         traj_ls.append(traj)

#     for id_, mask in lane_mask.items():
#         data = features[mask[0]+add_len: mask[1]+add_len]
#         lane = np.vstack((data[:, 0:2], data[-1, 2:4]))
#         lane_ls.append(lane)
    
#     return traj_ls, lane_ls
def coordinate_concat(xs, ys, vec_xs, vec_ys, idx):
    """ concat x, y into (x, y)"""
    x = xs[idx]
    y = ys[idx]
    vec_x = vec_xs[idx]
    vec_y = vec_ys[idx]
    xy = torch.hstack((x.reshape(-1,1),y.reshape(-1,1)))
    vec_xy = torch.hstack((vec_x.reshape(-1,1), vec_y.reshape(-1,1)))
    
    return xy, vec_xy

def reconstruct_polylines(features):
    """
    features: test_set._data.x, in the shape of [num_nodes, 6]
    """
    xs = features[:,0]
    ys = features[:,1]
    vec_xs = features[:,2]
    vec_ys = features[:,3]
    steps = features[:,4]
    poly_id = features[:,5]

    #plot cav
    idx0 = (poly_id == 0).nonzero(as_tuple=True)
    xy0, vec_xy0 = coordinate_concat(xs, ys, vec_xs, vec_ys, idx0)
    traj_end0 = xy0 + vec_xy0
    plt.plot(xy0[:,0], xy0[:,1], 'r')
    plt.plot(traj_end0[-2:,0], traj_end0[-2:,0], 'orange')

    #plot other vehicles and lanes
    for i in range(1, len(poly_id.unique())):
        idx = (poly_id == i).nonzero(as_tuple=True)
        xy, vec_xy = coordinate_concat(xs, ys, vec_xs, vec_ys, idx)
        step = steps[idx]
        if torch.max(step) == 0.0:  #lane centers
            lane_end = (xy*2+vec_xy)/2.0
            lane_start = (xy*2-vec_xy)/2.0
            plt.plot(lane_start[:,0], lane_start[:,1], '--', c='0.7')
            plt.plot(lane_end[-2:,0], lane_end[-2:,1], '--', c='0.7')
        else: #traj starts
            traj_end = xy + vec_xy
            plt.plot(xy[:,0], xy[:,1],'orange')
            plt.plot(traj_end[-2:,0], traj_end[-2:,0], 'orange')
    return traj_end0

def reconstruct_the_scene_with_predictions(features, pred_y, gt_y):
    plt.figure(dpi=200)
    cav_traj_end = reconstruct_polylines(features)
    cav_last_observed = cav_traj_end[-1,:]

    pred_y_cat = torch.vstack([pred_y[:, :50].reshape(-1, 1), pred_y[:, 50:].reshape(-1,1)])
    gt_y_cat = torch.vstack([gt_y[:, :50].reshape(-1,1), gt_y[:, 50:].reshape(-1,1)])
    pred_y_reconstruct = torch.vstack([pred_y_cat[0,:]+ cav_last_observed, pred_y_cat[1:,:]+pred_y_cat[:-1,:]+cav_last_observed])
    gt_y_reconstruct = torch.vstack([gt_y_cat[0,:]+cav_last_observed, gt_y_cat[1:,:]+gt_y_cat[:-1,:]+cav_last_observed])
    plt.plot(gt_y_reconstruct[:, 0], gt_y_reconstruct[:, 1], 'ro')
    plt.plot(pred_y_reconstruct[:, 0], pred_y_reconstruct[:, 1], color='b')#lw=0, marker='o', fillstyle="none")


def show_pred_and_gt(pred_y, gt_y, orig):
    """
    pred_y: torch.Tensor in shape of (1, 100)
    gt_y: torch.Tensor in shape of (1, 100)
    orig: torch.Tensor, last observed traj point in the shape of (1,2)
    """
    pred_y_cat = torch.vstack([pred_y[:, :50], pred_y[:, 50:]]).reshape(-1,2)
    gt_y_cat = torch.vstack([gt_y[:, :50], gt_y[:, 50:]]).reshape(-1,2)
    pred_y_devectornize = torch.vstack([pred_y_cat[0,:]+orig, pred_y_cat[1:,:]+pred_y_cat[:-1,:]+orig])
    gt_y_devectornize = torch.vstack([gt_y_cat[0,:]+orig, gt_y_cat[1:,:]+gt_y_cat[:-1,:]+orig])
    plt.plot(gt_y_devectornize[:, 0], gt_y_devectornize[:, 1], color='r')
    plt.plot(pred_y_devectornize[:, 0], pred_y_devectornize[:, 1], color='b')#lw=0, marker='o', fillstyle="none")

def show_predict_result(data, pred_y:torch.Tensor, y, add_len, show_lane=True):
    features, _ = data['POLYLINE_FEATURES'].values[0], data['GT'].values[0].astype(
        np.float32)
    traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
    lane_start, lane_end, traj_start, traj_end, cav_start, cav_end = reconstruct_polyline(features)

    type_ = 'CAV'
    for traj in traj_ls:
        show_traj(traj, type_)
        type_ = 'NCV'

    if show_lane:
        for lane in lane_ls:
            show_doubled_lane(lane)

    pred_y = pred_y.numpy().reshape((-1, 2)).cumsum(axis=0)
    y = y.numpy().reshape((-1, 2)).cumsum(axis=0)
    show_pred_and_gt(pred_y, y)