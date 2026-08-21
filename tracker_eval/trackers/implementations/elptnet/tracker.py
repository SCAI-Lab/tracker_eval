from .trajectory import Trajectory
from .box_op import *
import numpy as np

class Tracker3D:
    def __init__(self,tracking_features=False,
                    bb_as_features=False,
                    box_type='Kitti',
                    config = None,
                    output_coast_s=0.5,
                    association_gate_m=2.0):
        """
        initialize the the 3D tracker
        Args:
            tracking_features: bool, if tracking the features
            bb_as_features: bool, if tracking the bbs
            box_type: str, box type, available box type "OpenPCDet", "Kitti", "Waymo"
            output_coast_s: maximum duration for emitting unmatched predictions
            association_gate_m: maximum association cost/distance in metres
        """
        self.config = config
        self.current_timestamp = None
        self.current_pose = None
        self.current_bbs = None
        self.current_features = None
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.box_type = box_type
        self.output_coast_s = float(output_coast_s)
        self.association_gate_m = float(association_gate_m)
        if self.output_coast_s < 0:
            raise ValueError('output_coast_s must be non-negative')
        if self.association_gate_m <= 0:
            raise ValueError('association_gate_m must be positive')

        self.label_seed = 0

        self.active_trajectories = {}
        self.dead_trajectories = {}

    def tracking(self,bbs_3D = None,
                 features = None,
                 scores = None,
                 pose = None,
                 timestamp = None
                 ):
        """
        tracking the objects at the given timestamp
        Args:
            bbs: array(N,7) or array(N，7*k), 3D bounding boxes or 3D tracklets
                for tracklets, the boxes should be organized to [[box_t; box_t-1; box_t-2;...],...]
            features: array(N,k), the features of boxes or tracklets
            scores: array(N,), the detection score of boxes or tracklets
            pose: array(4,4), pose matrix to global scene
            timestamp: int, current timestamp, note that the timestamp should be consecutive

        Returns:
            bbs: array(M,7), the tracked bbs
            ids: array(M,), the assigned IDs for bbs
        """
        self.current_bbs = bbs_3D
        self.current_features = features
        self.current_scores = scores
        self.current_pose = pose
        self.current_timestamp = timestamp

        self.trajectores_prediction()

        if self.current_bbs is None:
            return self.collect_online_outputs()
        else:
            if len(self.current_bbs) == 0:
                return self.collect_online_outputs()

            else:
                self.current_bbs = convert_bbs_type(self.current_bbs,self.box_type)
                self.current_bbs = register_bbs(self.current_bbs,self.current_pose)
                ids = self.association()
                self.trajectories_update_init(ids)

                return self.collect_online_outputs()



    def trajectores_prediction(self):
        """
        predict the possible state of each active trajectories, if the trajectory is not updated for a while,
        it will be deleted from the active trajectories set, and moved to dead trajectories set
        Returns:

        """
        if len(self.active_trajectories) == 0 :
            return
        else:
            dead_track_id = []

            for key in self.active_trajectories.keys():
                if self.active_trajectories[key].consecutive_missed_num>=self.config.max_prediction_num:
                    dead_track_id.append(key)
                    continue
                if len(self.active_trajectories[key])-self.active_trajectories[key].consecutive_missed_num == 1 \
                    and len(self.active_trajectories[key])>= self.config.max_prediction_num_for_new_object :
                    dead_track_id.append(key)
                    continue
                self.active_trajectories[key].state_prediction(self.current_timestamp)

            for id in dead_track_id:
                tra = self.active_trajectories.pop(id)
                self.dead_trajectories[id]=tra

    def compute_cost_map(self):
        """
        compute the cost map between detections and predictions
        Returns:
              cost, array(N,M), where N is the number of detections, M is the number of active trajectories
              all_ids, list(M,), the corresponding IDs of active trajectories
        """
        all_ids = []

        all_predictions = []
        all_detections = []

        for key in self.active_trajectories.keys():
            all_ids.append(key)
            state = np.array(self.active_trajectories[key].trajectory[self.current_timestamp].predicted_state)
            state = state.reshape(-1)

            pred_score = np.array([self.active_trajectories[key].trajectory[self.current_timestamp].prediction_score])

            state = np.concatenate([state,pred_score])
            all_predictions.append(state)

        for i in range(len(self.current_bbs)):
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]
            label=1
            new_tra = Trajectory(init_bb=box,
                                 init_features=features,
                                 init_score=score,
                                 init_timestamp=self.current_timestamp,
                                 label=label,
                                 tracking_features=self.tracking_features,
                                 bb_as_features=self.bb_as_features,
                                 config = self.config)

            state = new_tra.trajectory[self.current_timestamp].predicted_state
            state = state.reshape(-1)
            all_detections.append(state)

        all_detections = np.array(all_detections)
        all_predictions = np.array(all_predictions)

        det_len = len(all_detections)
        pred_len = len(all_predictions)

        all_detections = all_detections.reshape((det_len,1,-1))
        all_predictions = all_predictions.reshape((1,pred_len,-1))

        all_detections = np.tile(all_detections,(1,pred_len,1))
        all_predictions = np.tile(all_predictions,(det_len,1,1))

        dis = (all_detections[...,0:3]-all_predictions[...,0:3])**2
        dis = np.sqrt(dis.sum(-1))

        cost = dis*all_predictions[...,-1]

        return cost,all_ids

    def association(self):
        """
        greedy assign the IDs for detected state based on the cost map
        Returns:
            ids, list(N,), assigned IDs for boxes, where N is the input boxes number
        """
        if len(self.active_trajectories) == 0:
            ids = []
            for i in range(len(self.current_bbs)):
                ids.append(self.label_seed)
                self.label_seed+=1
            return ids
        else:
            ids = []
            cost_map, all_ids = self.compute_cost_map()
            for i in range(len(self.current_bbs)):
                min = np.min(cost_map[i])
                arg_min = np.argmin(cost_map[i])

                if min < self.association_gate_m:
                    ids.append(all_ids[arg_min])
                    cost_map[:,arg_min] = 100000
                else:
                    ids.append(self.label_seed)
                    self.label_seed+=1
            return ids


    def trajectories_update_init(self,ids):
        """
        update a exiting trajectories based on the association results, or init a new trajectory
        Args:
            ids: list or array(N), the assigned ids for boxes
        """
        assert len(ids) == len(self.current_bbs)

        for i in range(len(self.current_bbs)):
            label = ids[i]
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]

            if label in self.active_trajectories.keys() and score>self.config.update_score:
                track = self.active_trajectories[label]
                track.state_update(
                     bb=box,
                     features=features,
                     score=score,
                     timestamp=self.current_timestamp)
            elif score>self.config.init_score:
                new_tra = Trajectory(init_bb=box,
                                     init_features=features,
                                     init_score=score,
                                     init_timestamp=self.current_timestamp,
                                     label=label,
                                     tracking_features=self.tracking_features,
                                     bb_as_features=self.bb_as_features,
                                     config = self.config)
                self.active_trajectories[label] = new_tra
            else:
                continue

    @staticmethod
    def _state_to_box(state):
        state = np.asarray(state).reshape(-1)
        if state.shape[0] < 13:
            raise RuntimeError('ELPTNet state does not contain box dimensions and yaw')
        return np.concatenate([state[0:3], state[9:13]]).astype(float)

    def _max_output_coast_frames(self):
        frequency = float(self.config.LiDAR_scanning_frequency)
        if frequency <= 0:
            raise ValueError('LiDAR_scanning_frequency must be positive')
        requested = int(np.ceil(self.output_coast_s * frequency))
        return min(requested, int(self.config.max_prediction_num))

    def collect_online_outputs(self):
        """Return detector-supported KF states and eligible short-gap predictions."""
        boxes = []
        ids = []
        max_coast_frames = self._max_output_coast_frames()
        post_score = float(getattr(self.config, 'post_score', 0.0))

        for track_id in sorted(self.active_trajectories.keys()):
            track = self.active_trajectories[track_id]
            current = track.trajectory.get(self.current_timestamp)
            if current is None:
                continue

            if current.updated_state is not None:
                score = current.score
                if score is None or float(score) < post_score:
                    continue
                boxes.append(self._state_to_box(current.updated_state))
                ids.append(track_id)
                continue

            if max_coast_frames <= 0 or not track.coast_eligible:
                continue
            if track.consecutive_missed_num > max_coast_frames:
                continue
            if current.predicted_state is None:
                continue

            boxes.append(self._state_to_box(current.predicted_state))
            ids.append(track_id)

        if not boxes:
            return np.zeros((0, 7), dtype=float), np.zeros((0,), dtype=int)
        return np.asarray(boxes, dtype=float), np.asarray(ids, dtype=int)


    def post_processing(self, config):
        """
        globally filter the trajectories
        Args:
            config: config

        Returns: dict(Trajectory)

        """
        tra = {}
        for key in self.dead_trajectories.keys():
            track = self.dead_trajectories[key]
            track.filtering(config)
            tra[key] = track
        for key in self.active_trajectories.keys():
            track = self.active_trajectories[key]
            track.filtering(config)
            tra[key] = track

        return tra


