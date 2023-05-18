import os
import sys
import numpy as np
import pandas as pd
import copy
from os.path import join as pjoin

sys.path.append('/Users/xichen/Documents/paper2-traj-pred/carla-data/maps/lanelet2')
import lane_segment, load_xml
sys.path.append('/Users/xichen/Documents/paper2-traj-pred/carla-data')
from utils.lane_sampling import Spline2D, visualize_centerline
import matplotlib.pyplot as plt

class scene_process():
    def __init__(self, split="train", obs_len=50, obs_range=50,
                 fut_len=50, cv_range=50, av_range=30,
                 normalized=True,save_dir=None, csv_folder="scene_mining"):
        self.COLOR_DICT = {"CAV": "#d33e4c", "CV": "g", "NCV": "darkorange"}
        self.split = split
        self.obs_len = obs_len
        self.obs_range = obs_range
        self.fut_len = fut_len
        self.cv_range = cv_range
        self.av_range = av_range
        self.normalized = normalized
        self.save_dir = save_dir
        self.abs_path = os.path.dirname(__file__)
        self.loader = CarlaDataLoader(pjoin(self.abs_path,"../../{}".format(csv_folder)))

    def __getitem__(self, idx, dir_post="train"):
        f_path = self.loader.seq_list[idx]
        df = pd.read_csv(f_path)
        path, seq_f_name_ext = os.path.split(f_path)

        return self.process_and_save(df, seq_id=idx, file_name=seq_f_name_ext, 
                                     dir_=pjoin(self.abs_path,"../../scene_mining_intermediate/",dir_post))
    
    def process_and_save(self, df, seq_id, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe: DataFrame, the data frame
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        df_processed = self.process(df, seq_id)
        self.save(df_processed, file_name, dir_)

    def save(self, df, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param df: DataFrame, the dataframe encoded
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        if not isinstance(df, pd.DataFrame):
            return

        if not dir_:
            dir_ = pjoin(os.path.split(self.root_dir)[0], "intermediate", self.split + "_intermediate", "raw")
        # else:
            # dir_ = os.path.join(dir_, self.split + "_intermediate", "raw")
           

        if not os.path.exists(dir_):
            os.makedirs(dir_)

        fname = f"{file_name}.pkl"
        df.to_pickle(os.path.join(dir_, fname))
      
    def get_map_polygon_bbox(self):
        rel_path = "../../maps/lanelet2/Town03.osm"
        roads = load_xml.load_lane_segments_from_xml(pjoin(self.abs_path, rel_path))
        polygon_bboxes, lane_starts, lane_ends = load_xml.build_polygon_bboxes(roads)
        self.roads = roads
        self.polygon_bboxes = polygon_bboxes
        self.lane_starts = lane_starts
        self.lane_ends = lane_ends

    def process(self, df, seq_id):
        data = self.read_scene_data(df)
        self.get_map_polygon_bbox()
        data = self.get_obj_feats(data)

        data['graph'] = self.get_lane_graph(data)
        data['seq_id'] = seq_id
        # visualization for debug purpose
        # self.visualize_data(data)
        return pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )

    def read_scene_data(self, df):
        """
        params: 
        df: dataframe from scene csv, with columns:
        frame, time, vid, type_id, position_x, position_y, position_z, 
        rotation_x, rotation_y, rotation_z, vel_x, vel_y, angular_z, 
        object_type, in_av_range

        return:
        data: dic with trajs and steps as keys. values in the order of [cav,ngbrs]
        """
        frames = np.sort(np.unique(df["frame"].values))
        mapping = dict()
        for i, frame in enumerate(frames):
            mapping[frame] = i
        
        trajs = np.concatenate((
                    df.position_x.to_numpy().reshape(-1, 1),
                    df.position_y.to_numpy().reshape(-1, 1)), 1)
        steps = [mapping[x] for x in df['frame'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['vid', 'object_type', 'in_av_range']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]
        
        cav_idx = obj_type.index("cav")
        idcs = objs[keys[cav_idx]]

        cav_traj = trajs[idcs]
        cav_step = steps[idcs]

        del keys[cav_idx]
        ngbr_trajs, ngbr_steps = [], []
        for key in keys:
            idcs = objs[key]
            ngbr_trajs.append(trajs[idcs])
            ngbr_steps.append(steps[idcs])

        obj_types = ['cav'] + [x[1] for x in keys]
        in_av_ranges = [True] + [x[2] for x in keys]

        data = dict()
        data["trajs"] = [cav_traj] + ngbr_trajs
        data["steps"] = [cav_step] + ngbr_steps
        data['obj_type'] = obj_types
        data["in_av_range"] = in_av_ranges

        return data
    
    def get_obj_feats(self, data):
        """
        params: 
        data: dic with trajs and steps as keys. values in the order of [cav,ngbrs]

        return:
        data: dic with normalized features
        """
        # get the origin and compute the oritentation of the target agent
        cav_orig = data['trajs'][0][self.obs_len-1].copy().astype(np.float32)
        # comput the rotation matrix
        if self.normalized:
            cav_heading_vector = cav_orig - data['trajs'][0][self.obs_len-2].copy().astype(np.float32)
            theta = np.arctan2(cav_heading_vector[1], cav_heading_vector[0])
            rot = np.asarray([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32)

        # get the target candidates and candidate gt
        cav_traj_obs = data['trajs'][0][0: self.obs_len].copy().astype(np.float32)
        cav_traj_fut = data['trajs'][0][self.obs_len:self.obs_len+self.fut_len].copy().astype(np.float32)

        query_x = cav_orig[0]
        query_y = cav_orig[1]
        ngbr_road_ids = load_xml.get_road_ids_in_xy_bbox(self.polygon_bboxes, self.lane_starts, self.lane_ends, self.roads, query_x, query_y, self.cv_range)


        # rotate the center lines and find the reference center line
        cav_traj_fut = np.matmul(rot, (cav_traj_fut - cav_orig.reshape(-1, 2)).T).T
        ctr_line_candts = []
        for i, _ in enumerate(ngbr_road_ids):
            road_id = ngbr_road_ids[i]
            ctr_line = np.stack(((self.roads[road_id].l_bound[:,0]+self.roads[road_id].r_bound[:,0])/2, 
                        (self.roads[road_id].l_bound[:,1]+self.roads[road_id].r_bound[:,1])/2),axis=-1)
            ctr_line_candts.append(np.matmul(rot, (ctr_line - cav_orig.reshape(-1, 2)).T).T)

        tar_candts = self.lane_candidate_sampling(ctr_line_candts, [0, 0], viz=False)

        if self.split == "test":
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, cav_traj_fut)
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, cav_traj_fut[-1])

        # cav_traj_obs_norm = np.matmul(rot, (cav_traj_obs - cav_orig.reshape(-1, 2)).T).T
        # plot_target_candidates(ctr_line_candts, cav_traj_obs_norm, cav_traj_fut, tar_candts)

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_len-1 not in step:
                continue

            #normalize and rotate
            traj_nd = np.matmul(rot, (traj - cav_orig.reshape(-1, 2)).T).T

            #collect the future prediction ground truth
            gt_pred = np.zeros((self.fut_len, 2), np.float32)
            has_pred = np.zeros(self.fut_len, np.bool)
            future_mask = np.logical_and(step >= self.obs_len, step < self.obs_len+self.fut_len)
            post_step = step[future_mask] - self.obs_len
            post_traj = traj_nd[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = True

            #collect the observation
            obs_mask = step < self.obs_len
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]

            for i in range(len(step_obs)):
                if step_obs[i] == self.obs_len - len(step_obs) + i:
                    break
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:]

            if len(step_obs) <= 1:
                continue

            feat = np.zeros((self.obs_len, 3), np.float32)
            has_obs = np.zeros(self.obs_len, np.bool)

            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            feats.append(feat)
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        has_obss = np.asarray(has_obss, np.bool)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data['cav_orig'] = cav_orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines

        return data
    
    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range.
        params: 
        data: dict with normalized features

        return:
        graph: dict, with keys "ctrs","feats","num_nodes","lane_idcs" 
                    representing centerpoint of a lane segment, 
                    length of lane vectors, 
                    total number of centerpoints,
                    lane id where the centerpoints come from
        """

        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        road_ids = load_xml.get_road_ids_in_xy_bbox(self.polygon_bboxes, self.lane_starts, self.lane_ends, self.roads, data['cav_orig'][0], data['cav_orig'][1], self.cv_range * 1.5)
        road_ids = copy.deepcopy(road_ids)

        lanes=dict()
        for road_id in road_ids:
            road = self.roads[road_id]
            ctr_line = np.stack(((self.roads[road_id].l_bound[:,0]+self.roads[road_id].r_bound[:,0])/2, 
                            (self.roads[road_id].l_bound[:,1]+self.roads[road_id].r_bound[:,1])/2),axis=-1)
            ctr_line = np.matmul(data["rot"], (ctr_line - data["cav_orig"].reshape(-1, 2)).T).T

            x, y = ctr_line[:,0], ctr_line[:,1]
            # if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
            #     continue
            # else:
            """getting polygons requires original centerline"""
            polygon, _, _ = load_xml.build_polygon_bboxes({road_id: self.roads[road_id]})
            polygon_x = np.array([polygon[:,0],polygon[:,0],polygon[:,2],polygon[:,2],polygon[:,0]])
            polygon_y = np.array([polygon[:,1],polygon[:,3],polygon[:,3],polygon[:,1],polygon[:,1]])
            polygon_reshape = np.concatenate([polygon_x,polygon_y],axis=-1) #shape(5,2)

            road.centerline = ctr_line
            road.polygon = np.matmul(data["rot"], (polygon_reshape - data['cav_orig'].reshape(-1, 2)).T).T
            lanes[road_id] = road

        lane_ids = list(lanes.keys())
        ctrs, feats = [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1]+ctrln[1:])/2.0, np.float32)) #lane center point
            feats.append(np.asarray(ctrln[1:]-ctrln[:-1], np.float32)) #length between waypoints

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i*np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count
        lane_idcs = np.concatenate(lane_idcs, 0)

        graph = dict()
        graph["ctrs"] = np.concatenate(ctrs, 0)
        graph["num_nodes"] = num_nodes
        graph["feats"] = np.concatenate(feats, 0)
        graph["lane_idcs"] = lane_idcs

        return graph
    
    # implement a candidate sampling with equal distance;
    def lane_candidate_sampling(self, centerline_list, orig=[0,0], distance=0.5, viz=False):
        """the input are list of lines, each line containing"""
        candidates = []
        for lane_id, line in enumerate(centerline_list):
            sp = Spline2D(x=line[:, 0], y=line[:, 1])
            s_o, d_o = sp.calc_frenet_position(np.array(line[0,0]), np.array(line[0,1]))
            s = np.arange(s_o, sp.s[-1], distance)
            ix, iy = sp.calc_global_position_online(s)
            candidates.append(np.stack([ix, iy], axis=1))
        candidates = np.unique(np.concatenate(candidates), axis=0)

        if viz:
            fig = plt.figure(0, figsize=(8, 7))
            fig.clear()
            for centerline_coords in centerline_list:
                visualize_centerline(centerline_coords)
            plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(centerline_list), len(candidates)))
            plt.show()

        return candidates
    
    def get_ref_centerline(self, cline_list, pred_gt):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx
        
    def get_candidate_gt(self, target_candidate, gt_target):
        """
        find the target candidate closest to the gt and output the one-hot ground truth
        :param target_candidate, (N, 2) candidates
        :param gt_target, (1, 2) the coordinate of final target
        """
        displacement = gt_target - target_candidate
        gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidate.shape[0], 1))
        onehot[gt_index] = 1

        offset_xy = gt_target - target_candidate[gt_index]
        return onehot, offset_xy
    
    def plot_target_candidates(self, candidate_centerlines, traj_obs, traj_fut, candidate_targets):
        fig = plt.figure(1, figsize=(8, 7))
        fig.clear()

        # plot centerlines
        for centerline_coords in candidate_centerlines:
            visualize_centerline(centerline_coords)

        # plot traj
        plt.plot(traj_obs[:, 0], traj_obs[:, 1], "x-", color="#d33e4c", alpha=1, linewidth=1, zorder=15)
        # plot end point
        plt.plot(traj_obs[-1, 0], traj_obs[-1, 1], "o", color="#d33e4c", alpha=1, markersize=6, zorder=15)
        # plot future traj
        plt.plot(traj_fut[:, 0], traj_fut[:, 1], "+-", color="b", alpha=1, linewidth=1, zorder=15)

        # plot target sample
        plt.scatter(candidate_targets[:, 0], candidate_targets[:, 1], marker="*", c="green", alpha=1, s=6, zorder=15)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(candidate_centerlines),
                                                                                        len(candidate_targets)))
        # plt.show(block=False)
        # plt.pause(0.01)
        plt.show()
    
    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_ctrs = data['graph']['ctrs']
        lines_feats = data['graph']['feats']
        lane_idcs = data['graph']['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss']
        preds = data['gt_preds']
        has_preds = data['has_preds']
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        # plt.show()
        plt.savefig('scene_process.png', dpi=fig.dpi)
        plt.show(block=False)
        plt.pause(5)
    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "CAV" if traj_id == 0 else "NCV"

        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], "d-", color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)

        plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))

        if len(pred) == 0:
            plt.text(obs[-1, 0], obs[-1, 1], "{}_e".format(traj_na))
        else:
            plt.text(pred[-1, 0], pred[-1, 1], "{}_e".format(traj_na))

class CarlaDataLoader():
    def __init__(self, root_dir):
        """ Load csv files from root_dir
        param: 
        root_dir: path to the folder containing sequence csv files
        """
        # count = 0
        # # Iterate directory
        # for path in os.listdir(self.file_dir):
        #     # check if current path is a file
        #     if os.path.isfile(pjoin(self.file_dir, path)):
        #         count += 1
        self.counter = 0
        self.seq_list = [pjoin(root_dir, x) for x in os.listdir(root_dir)]
        self.current_seq = self.seq_list[self.counter]

    def seq_df(self):
        """Get the dataframe for the current sequence."""

        return self.read_csv(self.current_seq)
    def read_csv(self, path):
        """csv reader
        params:
        path: Path to the csv file

        returns:
        dataframe containing the loaded csv
        """
        return pd.read_csv(path)

if __name__ == "__main__":
    frame = 29301
    cav = 615
    test = scene_process()
    rel_path = "../../scene_mining/scene_{}_{}".format(frame,cav)
    
    df = pd.read_csv(os.path.join(test.abs_path, rel_path))
    
    data = test.read_scene_data(df)
    test.get_map_polygon_bbox()
    data = test.get_obj_feats(data)
    data['graph'] = test.get_lane_graph(data)
    # out = pd.DataFrame(
    #         [[data[key] for key in data.keys()]],
    #         columns=[key for key in data.keys()]
    #     )
    test.visualize_data(data)

    
