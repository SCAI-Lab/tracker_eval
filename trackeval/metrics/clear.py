
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils

class CLEAR(_BaseMetric):
    """Class which implements the CLEAR metrics"""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a TP match. Default 0.5.
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
            'RETURN_PER_GT': False,
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        main_integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
        extra_integer_fields = ['CLR_Frames']
        self.integer_fields = main_integer_fields + extra_integer_fields
        main_float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA']
        extra_float_fields = ['CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']
        self.float_fields = main_float_fields + extra_float_fields
        self.fields = self.float_fields + self.integer_fields
        self.summed_fields = self.integer_fields + ['MOTP_sum']
        self.summary_fields = main_float_fields + main_integer_fields

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])
        self.return_per_gt = bool(self.config['RETURN_PER_GT'])


    @_timing.time
    def eval_sequence(self, data):
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['CLR_FN'] = data['num_gt_dets']
            res['ML'] = data['num_gt_ids']
            res['MLR'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['CLR_FP'] = data['num_tracker_dets']
            res['MLR'] = 1.0
            return res

        # Variables counting global association
        num_gt_ids = data['num_gt_ids']
        gt_id_count = np.zeros(num_gt_ids, dtype=np.int32)
        gt_matched_count = np.zeros(num_gt_ids, dtype=np.int32)
        gt_frag_count = np.zeros(num_gt_ids, dtype=np.int32)  # also better as int
        gt_idsw_count = np.zeros(num_gt_ids, dtype=np.int32)     # per-GT ID switches
        gt_motp_sum = np.zeros(num_gt_ids, dtype=np.float64)     # per-GT sum IoU over matched frames

        # --- NEW: event-conditioned GT stats at IDSW frames (per GT id index) ---
        gt_idsw_dens_sum = np.zeros(num_gt_ids, dtype=np.float64)
        gt_idsw_dens_cnt = np.zeros(num_gt_ids, dtype=np.int32)

        gt_idsw_nn_sum = np.zeros(num_gt_ids, dtype=np.float64)
        gt_idsw_nn_cnt = np.zeros(num_gt_ids, dtype=np.int32)

        # Fetch per-frame arrays if dataset provided them
        frame_dens_event = data.get('gt_frame_density_count_event', None)  # list[t] -> vector aligned to gt_ids_t
        frame_nn = data.get('gt_frame_nn_dist', None)                      # list[t] -> vector aligned to gt_ids_t

        # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
        # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
        prev_tracker_id = np.nan * np.zeros(num_gt_ids)  # For scoring IDSW
        prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)  # For matching IDSW

        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            #print(t,gt_ids_t, tracker_ids_t)
            # print(asdasd)
            if len(gt_ids_t) == 0:
                res['CLR_FP'] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                res['CLR_FN'] += len(gt_ids_t)
                gt_id_count[gt_ids_t] += 1
                continue

            # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily
            similarity = data['similarity_scores'][t]
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < self.threshold - np.finfo('float').eps] = 0
            #print('before ass',score_mat)
            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            #print('matched',match_rows, match_cols)
            actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = gt_ids_t[match_rows]
            #print(matched_gt_ids)
            matched_tracker_ids = tracker_ids_t[match_cols]

            # Calc IDSW for MOTA
            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
                np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))
            res['IDSW'] += np.sum(is_idsw)
            if self.return_per_gt and len(matched_gt_ids) > 0:
                gt_idsw_count[matched_gt_ids[is_idsw]] += 1
            
            # --- NEW: accumulate density/NN stats at the exact frames where IDSW occurs ---
            if self.return_per_gt and np.any(is_idsw):
                # match_rows are indices into gt_ids_t for the matched GT detections
                # rows_idsw are the matched GT det row indices at switch frames
                rows_idsw = match_rows[is_idsw]
                gt_ids_idsw = matched_gt_ids[is_idsw]

                if frame_dens_event is not None and frame_dens_event[t] is not None and len(frame_dens_event[t]) > 0:
                    dens_vals = frame_dens_event[t][rows_idsw]
                    gt_idsw_dens_sum[gt_ids_idsw] += dens_vals.astype(np.float64)
                    gt_idsw_dens_cnt[gt_ids_idsw] += 1

                if frame_nn is not None and frame_nn[t] is not None and len(frame_nn[t]) > 0:
                    nn_vals = frame_nn[t][rows_idsw]
                    # ignore nan nn distances
                    ok = np.isfinite(nn_vals)
                    if np.any(ok):
                        gt_idsw_nn_sum[gt_ids_idsw[ok]] += nn_vals[ok].astype(np.float64)
                        gt_idsw_nn_cnt[gt_ids_idsw[ok]] += 1

            # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestep
            gt_id_count[gt_ids_t] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)
            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
            currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

            # Calculate and accumulate basic statistics
            num_matches = len(matched_gt_ids)
            #print(len(gt_ids_t))
            res['CLR_TP'] += num_matches
            res['CLR_FN'] += len(gt_ids_t) - num_matches
            res['CLR_FP'] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                #print(sum(similarity[match_rows, match_cols]))
                # match_sims = 1-similarity[match_rows, match_cols]
                match_sims = similarity[match_rows, match_cols]
                res['MOTP_sum'] += sum(match_sims)

                if self.return_per_gt:
                    gt_motp_sum[matched_gt_ids] += match_sims

        # Calculate MT/ML/PT/Frag/MOTP
        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res['MT'] = np.sum(np.greater(tracked_ratio, 0.8))
        res['PT'] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - res['MT']
        res['ML'] = num_gt_ids - res['MT'] - res['PT']
        res['Frag'] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1))
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])

        res['CLR_Frames'] = data['num_timesteps']

        # Calculate final CLEAR scores
        res = self._compute_final_fields(res)
        if self.return_per_gt:
            gt_orig_ids = data.get('gt_orig_ids', np.arange(num_gt_ids))

            tracked_ratio_all = np.zeros(num_gt_ids, dtype=np.float64)
            mask = gt_id_count > 0
            tracked_ratio_all[mask] = gt_matched_count[mask] / gt_id_count[mask]

            gt_mt = tracked_ratio_all > 0.8
            gt_pt = (tracked_ratio_all >= 0.2) & (~gt_mt)
            gt_ml = (~gt_mt) & (~gt_pt)

            gt_frag = np.maximum(0, gt_frag_count - 1).astype(np.int32)

            gt_motp = np.full(num_gt_ids, np.nan, dtype=np.float64)
            motp_mask = gt_matched_count > 0
            gt_motp[motp_mask] = gt_motp_sum[motp_mask] / gt_matched_count[motp_mask]


            res['_per_gt'] = {
                'gt_id': gt_orig_ids.astype(np.int64),
                'frames': gt_id_count.astype(np.int32),
                'tp': gt_matched_count.astype(np.int32),
                'fn': (gt_id_count - gt_matched_count).astype(np.int32),
                'idsw': gt_idsw_count.astype(np.int32),
                'frag': gt_frag,
                'tracked_ratio': tracked_ratio_all,
                'mt': gt_mt.astype(np.int8),
                'pt': gt_pt.astype(np.int8),
                'ml': gt_ml.astype(np.int8),
                'motp_sum': gt_motp_sum,
                'motp': gt_motp,
            }

            gt_avg_dist = data.get('gt_avg_distance', None)
            gt_avg_dens = data.get('gt_avg_density', None)
            gt_avg_speed = data.get('gt_avg_speed_savgol_mps', None)
            gt_move_frac = data.get('gt_move_fraction_vgt_thresh', None)
            gt_move_thresh = data.get('gt_speed_move_thresh', None)

            if gt_avg_speed is not None:
                res['_per_gt']['avg_speed_savgol_mps'] = gt_avg_speed.astype(np.float64)
            if gt_move_frac is not None:
                # keep the threshold in the name stable; store actual threshold separately too
                res['_per_gt']['move_fraction_vgt_thresh'] = gt_move_frac.astype(np.float64)
            if gt_move_thresh is not None:
                res['_per_gt']['move_thresh_mps'] = float(gt_move_thresh)


            if gt_avg_dist is not None:
                res['_per_gt']['avg_distance'] = gt_avg_dist.astype(np.float64)
            if gt_avg_dens is not None:
                res['_per_gt']['avg_density_r2m'] = gt_avg_dens.astype(np.float64)

            # --- NEW: attach extra GT-derived exposure fields if present in `data` ---
            # These were computed in JRDB3DBox.get_preprocessed_seq_data()

            # NN stats
            gt_nn_mean = data.get('gt_nn_mean_m', None)
            gt_nn_p05  = data.get('gt_nn_p05_m', None)
            gt_nn_min  = data.get('gt_nn_min_m', None)
            if gt_nn_mean is not None: res['_per_gt']['nn_dist_mean_m'] = gt_nn_mean.astype(np.float64)
            if gt_nn_p05  is not None: res['_per_gt']['nn_dist_p05_m']  = gt_nn_p05.astype(np.float64)
            if gt_nn_min  is not None: res['_per_gt']['nn_dist_min_m']  = gt_nn_min.astype(np.float64)

            # NN threshold fractions (if present)
            for key in list(data.keys()):
                if key.startswith('gt_nn_frac_lt'):
                    res['_per_gt'][key.replace('gt_', '')] = data[key].astype(np.float64)

            # Density stats at radii (if present)
            for key in list(data.keys()):
                if key.startswith('gt_dens_r') and (key.endswith('_mean') or key.endswith('_p95') or key.endswith('_max') or '_frac_ge' in key):
                    res['_per_gt'][key.replace('gt_', '')] = data[key].astype(np.float64)

            # Range percentiles (optional)
            if data.get('gt_avg_distance_p05', None) is not None:
                res['_per_gt']['avg_distance_p05'] = data['gt_avg_distance_p05'].astype(np.float64)
            if data.get('gt_avg_distance_p95', None) is not None:
                res['_per_gt']['avg_distance_p95'] = data['gt_avg_distance_p95'].astype(np.float64)

            # Matchable fraction/count
            if data.get('gt_matchable_count', None) is not None:
                res['_per_gt']['matchable_count'] = data['gt_matchable_count'].astype(np.int32)
            if data.get('gt_matchable_frac', None) is not None:
                res['_per_gt']['matchable_frac'] = data['gt_matchable_frac'].astype(np.float64)
            if data.get('gt_matchable_thr', None) is not None:
                res['_per_gt']['matchable_thr'] = float(data['gt_matchable_thr'])

            # --- NEW: IDSW event-conditioned means (computed in CLEAR above) ---
            idsw_dens_mean = np.full(num_gt_ids, np.nan, dtype=np.float64)
            m = gt_idsw_dens_cnt > 0
            idsw_dens_mean[m] = gt_idsw_dens_sum[m] / gt_idsw_dens_cnt[m]
            res['_per_gt']['idsw_density_count_event_mean'] = idsw_dens_mean
            res['_per_gt']['idsw_event_count'] = gt_idsw_dens_cnt.astype(np.int32)

            idsw_nn_mean = np.full(num_gt_ids, np.nan, dtype=np.float64)
            m = gt_idsw_nn_cnt > 0
            idsw_nn_mean[m] = gt_idsw_nn_sum[m] / gt_idsw_nn_cnt[m]
            res['_per_gt']['idsw_nn_dist_mean_m'] = idsw_nn_mean

        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0}, field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                res[field] = np.mean(
                    [v[field] for v in all_res.values() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0], axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        num_gt_ids = res['MT'] + res['ML'] + res['PT']
        res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
        res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
        res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
        res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FP'])
        res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])

        res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5*res['CLR_FN'] + 0.5*res['CLR_FP'])
        res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
        safe_log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
        res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        return res
