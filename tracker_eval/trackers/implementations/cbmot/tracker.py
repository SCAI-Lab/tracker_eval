import copy

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
    'construction_vehicle',
    'barrier',
    'traffic_cone',
]

# 99.9 percentile of the l2 velocity error distribution (per class / 0.5 second)
# This is an earlier statistics and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
    'car': 3,
    'truck': 4,
    'bus': 5.5,
    'trailer': 2,
    'pedestrian': 0.7,
    'motorcycle': 4,
    'bicycle': 2.5,
    'construction_vehicle': 1,
    'barrier': 1,
    'traffic_cone': 1,
}


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)


def comparing_positions(self, positions1_data, positions2_data, positions1, positions2):
    M = len(positions1_data)
    N = len(positions2_data)

    positions1_cat = np.array([x['label_preds'] for x in positions1_data], np.int32)
    positions2_cat = np.array([x['label_preds'] for x in positions2_data], np.int32)
    max_diff = np.array([self.velocity_error[x['detection_name']] for x in positions2_data], np.float32)

    if len(positions1) > 0:
        dist = (((positions1.reshape(1, -1, 2) - positions2.reshape(-1, 1, 2)) ** 2).sum(axis=2))
        dist = np.sqrt(dist)

        invalid = ((dist > max_diff.reshape(N, 1)) +
                   (positions2_cat.reshape(N, 1) != positions1_cat.reshape(1, M))) > 0
        dist = dist + invalid * 1e18

        if self.hungarian:
            dist[dist > 1e18] = 1e18
            matched_indices = linear_sum_assignment(copy.deepcopy(dist))
            matched_indices = reshape(matched_indices)  # <-- critical
        else:
            matched_indices = greedy_assignment(copy.deepcopy(dist))
    else:
        assert M == 0
        matched_indices = np.array([], np.int32).reshape(-1, 2)

    unmatched_positions1_data = [d for d in range(positions1.shape[0]) if d not in matched_indices[:, 1]]
    unmatched_positions2_data = [d for d in range(positions2.shape[0]) if d not in matched_indices[:, 0]]

    if self.hungarian:
        matches = []
        for m in matched_indices:
            if dist[m[0], m[1]] > 1e16:
                unmatched_positions2_data.append(m[0])  # det unmatched
                unmatched_positions1_data.append(m[1])  # track unmatched (important)
            else:
                matches.append(m)

        unmatched_positions1_data = sorted(set(unmatched_positions1_data))  # <-- add
        unmatched_positions2_data = sorted(set(unmatched_positions2_data))  # <-- add
        matches = np.array(matches, dtype=np.int32).reshape(-1, 2)
    else:
        matches = matched_indices

    return matches, unmatched_positions1_data, unmatched_positions2_data



def update_function(self, track, m, loaded_model):
    if self.score_update == 'nn':
        import torch
        inp = np.array([self.tracks[m[1]]['detection_score'], track['detection_score']])
        new = loaded_model(torch.Tensor(inp))
        track['detection_score'] = new.item()

    elif self.score_update == 'parallel_addition':
        track['detection_score'] = 1 - ((1 - track['detection_score']) * (
                1 - self.tracks[m[1]]['detection_score'])) / (
                                           (1 - track['detection_score']) + (
                                           1 - self.tracks[m[1]]['detection_score']))
    elif self.score_update == 'multiplication':
        track['detection_score'] = 1 - (1 - track['detection_score']) * \
                                   (1 - self.tracks[m[1]]['detection_score'])
    elif self.score_update == 'addition':
        track['detection_score'] += self.tracks[m[1]]['detection_score']
        track['detection_score'] = np.clip(track['detection_score'], a_min=0.0, a_max=1.0)
    elif self.score_update == 'max':
        track['detection_score'] = np.maximum(track['detection_score'],
                                              self.tracks[m[1]]['detection_score'])

    return track


# comparing the results of matching the det with trk and compare them with ground truth resulting train data
def tain_data(self, annotated_data, first_matching_detections, match1_matched_det_pos, match1_matched_trk):
    # prepare the ground truth data for matching
    ground_truth = annotated_data['anns']
    ground_truth_pos = np.array([index['translation'][:2] for index in ground_truth])
    # matching the matched and the ground truth
    training_matching = comparing_positions(self, first_matching_detections, ground_truth, match1_matched_det_pos,
                                            ground_truth_pos)
    # get the sets from the matching (only two are needed)
    matched_training_matching, unmatched_training_matching, unmatched_ground_truth = training_matching[0], training_matching[1], training_matching[2]

    # PS: false positive : the reference is the system and not the ground truth
    # PS: True positive have the label 1 and false positive 0

    # get matched detections (true positive)
    det_target_1 = get_data_from_index(matched_training_matching, 1, first_matching_detections)
    # get unmatched detections (false positive)
    det_target_0 = get_data_from_index(np.array(unmatched_training_matching), 1, first_matching_detections)
    # get matched tracklets (true positive)
    trk_target_1 = get_data_from_index(matched_training_matching, 1, match1_matched_trk)
    # get unmatched tracklets (false positive)
    trk_target_0 = get_data_from_index(np.array(unmatched_training_matching), 1, match1_matched_trk)

    # concatenate detection's detection_score [det_sc : True positive,,,det_sc : false positive,,,]
    det_target_1_0 = np.array(
        [index['detection_score'] for index in np.concatenate((det_target_1, det_target_0))])
    # concatenate tracklet's detection_score [tr_sc : True positive,,,tr_sc : false positive,,,]
    trk_target_1_0 = np.array(
        [index['detection_score'] for index in np.concatenate((trk_target_1, trk_target_0))])
    # get the two score inputs lists [[tr_sc,det_sc],,,]
    inputs = list(
        [[trk_target_1_0[index], det_target_1_0[index]]
         for index in range(len(det_target_1_0))])
    # get the labels list
    labels = list(np.concatenate(
        (np.ones(len(det_target_1), np.float64), np.zeros(len(det_target_0), np.float64))))
    # each iteration return a dictionary with the inputs and labels
    train_set = {'input': inputs, 'labels': labels}
    return train_set


def get_data_from_index(index_list, i, data_source):
    if len(index_list.shape) == 1:
        return np.array([data_source[index] for index in index_list])

    else:
        return np.array([data_source[index] for index in np.array([match[i] for match in index_list])])


WAYMO_TRACKING_NAMES = [
    1,
    2,
    4,
]
WAYMO_CLS_VELOCITY_ERROR = {
    1: 2,
    2: 0.2,
    4: 0.5,
}


# reshape hungarians output to match the greedy output shape
def reshape(hungarian):
    result = np.empty((0, 2), int)
    for i in range(len(hungarian[0])):
        result = np.append(result, np.array([[hungarian[0][i], hungarian[1][i]]]), axis=0)
    return result


class PubTracker(object):
    def __init__(self, hungarian=False, max_age=6, noise=0.05, active_th=1, min_hits=1, score_update=None,
                 deletion_th=0.0,
                 detection_th=0.0, dataset='Nuscenes', model_path='LeakyReLU.th',
                 motion_model='PointTracker', use_vel=False,
                 distance_gate_m=None, score_decay_per_s=None):
        if motion_model not in ('PointTracker', 'KF'):
            raise ValueError("motion_model must be 'PointTracker' or 'KF'")
        self.tracker = motion_model
        self.hungarian = hungarian
        self.max_age = max_age
        self.min_hits = min_hits
        self.noise = noise
        self.score_decay_per_s = score_decay_per_s
        self.s_th = active_th  # activate threshold
        self.score_update = score_update
        self.det_th = detection_th  # detection threshold
        self.del_th = deletion_th  # deletion threshold
        self.use_vel = use_vel

        if score_update == 'nn':
            import torch
            self.loaded_model = torch.load(model_path)
            self.loaded_model.eval()
        else:
            self.loaded_model = None

        print("Use hungarian: {}".format(hungarian))

        if dataset == 'Nuscenes':
            self.velocity_error = NUSCENE_CLS_VELOCITY_ERROR
            self.tracking_names = NUSCENES_TRACKING_NAMES
        elif dataset == 'Waymo':
            self.velocity_error = WAYMO_CLS_VELOCITY_ERROR
            self.tracking_names = WAYMO_TRACKING_NAMES
        else:
            raise ValueError("dataset must be 'Nuscenes' or 'Waymo'")

        self.velocity_error = dict(self.velocity_error)
        if distance_gate_m is not None:
            gate = float(distance_gate_m)
            if gate <= 0:
                raise ValueError('distance_gate_m must be positive')
            self.velocity_error = {name: gate for name in self.tracking_names}
        self.id_count = 0
        self.tracks = []

        self.reset()

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def _score_decay(self, time_lag):
        if self.score_decay_per_s is None:
            return float(self.noise)
        return max(0.0, float(self.score_decay_per_s) * float(time_lag))

    def _init_kf(self, track):
        dim_z = 4 if self.use_vel else 2
        track['KF'] = KalmanFilter(6, dim_z)
        if self.use_vel:
            track['KF'].H = np.array([
                [1., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.],
            ])
        else:
            track['KF'].H = np.array([
                [1., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0.],
            ])
        velocity = np.asarray(track.get('velocity', [0.0, 0.0])[:2], dtype=float)
        track['KF'].x = np.hstack([track['ct'], velocity, np.zeros(2)])
        track['KF'].P *= 10

    @staticmethod
    def _sync_kf_geometry(track):
        state = np.asarray(track['KF'].x).reshape(-1)
        track['ct'] = state[:2].astype(float)
        track['translation'][0] = float(state[0])
        track['translation'][1] = float(state[1])
        if 'velocity' not in track or len(track['velocity']) < 2:
            track['velocity'] = [0.0, 0.0, 0.0]
        track['velocity'][0] = float(state[2])
        track['velocity'][1] = float(state[3])

    def _new_track(self, result, first_frame=False):
        track = result
        self.id_count += 1
        track['tracking_id'] = self.id_count
        track['age'] = 1
        if first_frame:
            # Preserve the released sequence-start behavior.
            track['active'] = self.min_hits
        else:
            track['active'] = 1 if track['detection_score'] >= self.det_th else 0
        if self.tracker == 'KF':
            self._init_kf(track)
            self._sync_kf_geometry(track)
        return track

    def step_centertrack(self, results, annotated_data, time_lag, version, train_data):
        """
        computes connections between current resources with resources from older frames
        :param results: resources in one specific frame
        :param annotated_data: ground truth for train data
        :param time_lag: time between two successive frame (difference in their timestamp)
        :param version: trainval or test
        :param train_data: boolean true if train_data needed false else
        :param model_path: model_path for learning score update function
        :return: tracks: tracklets (detection + tracking id, age, activity) for one specific frame
                 if train_data true than also return the training data
        """

        time_lag = float(time_lag)
        if time_lag <= 0:
            raise ValueError('time_lag must be positive')
        decay = self._score_decay(time_lag)

        temp = []
        for det in results:
            if det['detection_name'] not in self.tracking_names:
                continue
            det['ct'] = np.array(det['translation'][:2], dtype=float)
            if self.tracker == 'PointTracker':
                det['tracking'] = np.array(det['velocity'][:2], dtype=float) * -time_lag
            det['label_preds'] = self.tracking_names.index(det['detection_name'])
            temp.append(det)
        results = temp

        N = len(results)  # number of resources in this frame
        M = len(self.tracks)  # number of tracklets
        ret = []  # initiate return value (will become the updated tracklets list)

        # if no tracklet exist just yet (i.e. processing the first frame)
        if M == 0:
            for result in results:
                ret.append(self._new_track(result, first_frame=True))
            self.tracks = ret
            if train_data:
                return ret, {}
            else:
                return ret

        # Processing from the second frame
        if self.tracker == 'PointTracker':
            if len(results) and 'tracking' in results[0]:
                dets = np.array(
                    [det['ct'].astype(np.float32) + det['tracking'].astype(np.float32)
                     for det in results], np.float32)
            else:
                dets = np.asarray([det['ct'] for det in results], np.float32).reshape(-1, 2)

            tracks = np.array(
                [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        elif self.tracker == 'KF':
            dets = np.asarray([det['ct'] for det in results], np.float32).reshape(-1, 2)

            tracks = []
            for tracklet in self.tracks:
                tracklet['KF'].predict(F=np.array([[1, 0, time_lag, 0, time_lag * time_lag, 0],
                                                   [0, 1, 0, time_lag, 0, time_lag * time_lag],
                                                   [0, 0, 1, 0, time_lag, 0],
                                                   [0, 0, 0, 1, 0, time_lag],
                                                   [0, 0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 0, 1]]))
                self._sync_kf_geometry(tracklet)
                tracks.append(np.asarray(tracklet['KF'].x[:2]).reshape(-1))

            tracks = np.array(tracks, np.float32)  # M x 2

        # matching the current with the estimated pass
        matching = comparing_positions(self, self.tracks, results, tracks, dets)
        matched, unmatched_trk, unmatched_det = matching[0], matching[1], matching[2]

        # train data for update function
        if version == 'v1.0-trainval' and train_data:
            matched_det = get_data_from_index(matched, 0, results)
            matched_det_pos = get_data_from_index(matched, 0, dets)
            matched_trk = get_data_from_index(matched, 1, self.tracks)
            train_set = tain_data(self, annotated_data, matched_det, matched_det_pos, matched_trk)

        # add matches
        for m in matched:
            # initiate new tracklet (with three additional attributes)
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']  # tracklet id = id of matched trackled
            track['age'] = 1  # how many frames without matching detection (i.e. inactivity)
            track['active'] = self.tracks[m[1]]['active'] + 1
            if self.tracker == 'KF':
                track['KF'] = self.tracks[m[1]]['KF']
                if self.use_vel:
                    track['KF'].update(z=np.hstack([track['ct'], np.array(track['velocity'][:2])]))
                else:
                    track['KF'].update(z=track['ct'])
                self._sync_kf_geometry(track)
            self.tracks[m[1]]['detection_score'] = np.clip(self.tracks[m[1]]['detection_score'] - decay,
                                                           a_min=0.0, a_max=1.0)
            # update detection score
            track['detection_score'] = update_function(self, track, m, self.loaded_model)['detection_score']
            ret.append(track)

        # add unmatched resources as new 'born' tracklets
        for i in unmatched_det:
            ret.append(self._new_track(results[i]))

        # still store unmatched tracks, however, we shouldn't output the object in current frame
        for i in unmatched_trk:
            track = self.tracks[i]

            # update score (only apply score decay)
            if self.score_update is not None:
                track['detection_score'] = float(np.clip(
                    track['detection_score'] - decay, a_min=0.0, a_max=1.0
                ))

            # keep tracklet if score is above threshold AND age is not too high
            if track['age'] < self.max_age and track['detection_score'] > self.del_th:
                track['age'] += 1
                if track['detection_score'] > self.s_th:
                    track['active'] += 1
                else:
                    track['active'] = 0
                ct = track['ct']

                if 'tracking' in track:
                    offset = track['tracking'] * -1  # move forward
                    track['ct'] = ct + offset
                    track['translation'][0] = float(track['ct'][0])
                    track['translation'][1] = float(track['ct'][1])
                elif self.tracker == 'KF':
                    self._sync_kf_geometry(track)
                ret.append(track)

        self.tracks = ret
        if train_data:
            return ret, train_set
        else:
            return ret
