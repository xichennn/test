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

def visualize_centerline(centerline) -> None:
    """Visualize the computed centerline.
    Args:
        centerline: Sequence of coordinates forming the centerline
    """
    line_coords = list(zip(*centerline))
    lineX = line_coords[0]
    lineY = line_coords[1]
    plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
    plt.text(lineX[0], lineY[0], "s")
    plt.text(lineX[-1], lineY[-1], "e")
    plt.axis("equal")

def show_traj(traj, type_):
    """
    args: ndarray in shape of (n, 2)
    """
    plt.plot(traj[:, 0], traj[:, 1], color=COLOR_DICT[type_])

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
    category = features[:,5]
    poly_id = features[:,6]

    #visualize cav
    idx0 = (poly_id == 0).nonzero(as_tuple=True)
    xy0, vec_xy0 = coordinate_concat(xs, ys, vec_xs, vec_ys, idx0)
    traj_end0 = xy0 + vec_xy0
    plt.plot(xy0[:,0], xy0[:,1], 'r')
    plt.plot(traj_end0[-2:,0], traj_end0[-2:,1], 'r')

    #visualize other vehicles and lanes
    for i in range(1, len(poly_id.unique())):
        idx = (poly_id == i).nonzero(as_tuple=True)
        xy, vec_xy = coordinate_concat(xs, ys, vec_xs, vec_ys, idx)
        step = steps[idx]
        if torch.max(step) == 0.0:  #lane centers
            lane_start = (xy*2.0-vec_xy)/2.0
            lane_end = (xy[-1,:]*2.0+vec_xy[-1,:])/2.0
            lane = np.vstack([lane_start, lane_end.reshape(-1, 2)])
            # plt.plot(lane_start[:,0], lane_start[:,1], '--', c='0.7', zorder=0)
            # plt.plot(lane_end[-2:,0], lane_end[-2:,1], '--', c='0.7', zorder=0)
            visualize_centerline(lane)
        else: #traj starts
            traj_end = xy + vec_xy
            traj = np.vstack([xy, traj_end[-1,:].reshape(-1, 2)])
            # plt.plot(traj[:,0], traj[:,1],'orange', zorder=5)
            plt.plot(traj[:, 0], traj[:, 1], color=COLOR_DICT["NCV"], alpha=1, linewidth=1, zorder=5)
            plt.text(traj[0, 0], traj[0, 1], "{}_s".format(i), c='darkorange')
            # plot_traj(traj,torch.zeros_like(traj))
    return traj_end0[-1,:]

def reconstruct_the_scene_with_predictions(features, pred_y, gt_y):
    """
    pred_y: torch.Tensor in shape of (50, 2)
    gt_y: torch.Tensor in shape of (50, 2)

    """
    plt.figure(dpi=200)
    cav_last_observed = reconstruct_polylines(features)

    # reconstruct y from offset
    pred_y_ = [list(torch.sum(pred_y[:i,:],axis=0)) for i in range(pred_y.shape[0])]
    pred_y_reconstruct = torch.FloatTensor(pred_y_) + cav_last_observed

    gt_y_ = [list(torch.sum(gt_y[:i,:],axis=0)) for i in range(gt_y.shape[0])]
    gt_y_reconstruct = torch.FloatTensor(gt_y_) + cav_last_observed

    plt.plot(gt_y_reconstruct[:, 0], gt_y_reconstruct[:, 1], "--", color=COLOR_DICT["CAV"], alpha=1, linewidth=1, zorder=5)
    plt.plot(pred_y_reconstruct[:, 0], pred_y_reconstruct[:, 1], color='b', zorder=5)#lw=0, marker='o', fillstyle="none")

def show_pred_and_gt(pred_y, gt_y, orig):
    """
    pred_y: torch.Tensor in shape of (100,)
    gt_y: torch.Tensor in shape of (100,)
    orig: torch.Tensor, last observed traj point in the shape of (1,2)
    """
    pred_y_cat = pred_y.reshape(-1,2)
    gt_y_cat = gt_y.reshape(-1,2)
    pred_y_ = [list(torch.sum(pred_y_cat[:i,:],axis=0)) for i in range(pred_y_cat.shape[0])]
    pred_y_reconstruct = torch.FloatTensor(pred_y_) + orig
    gt_y_ = [list(torch.sum(gt_y_cat[:i,:],axis=0)) for i in range(gt_y_cat.shape[0])]
    gt_y_reconstruct = torch.FloatTensor(gt_y_) + orig
    plt.plot(gt_y_reconstruct[:, 0], gt_y_reconstruct[:, 1], color='r')
    plt.plot(pred_y_reconstruct[:, 0], pred_y_reconstruct[:, 1], color='b')#lw=0, marker='o', fillstyle="none")

def show_predict_result(data, pred_y:torch.Tensor, y, add_len, show_lane=True):
    features, _ = data['POLYLINE_FEATURES'].values[0], data['GT'].values[0].astype(
        np.float32)
    traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
    lane_start, lane_end, traj_start, traj_end, cav_start, cav_end = reconstruct_polylines(features)

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
