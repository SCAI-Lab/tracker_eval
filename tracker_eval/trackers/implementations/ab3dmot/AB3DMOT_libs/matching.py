import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from AB3DMOT_libs.dist_metrics import iou, dist3d, dist_ground, m_distance

def compute_affinity(dets, trks, metric, trk_inv_inn_matrices=None):
	# compute affinity matrix

	aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
	for d, det in enumerate(dets):
		for t, trk in enumerate(trks):

			# choose to use different distance metrics
			if 'iou' in metric:       dist_now = iou(det, trk, metric)            
			elif metric == 'm_dis':   dist_now = -m_distance(det, trk, trk_inv_inn_matrices[t])
			elif metric == 'euler':   dist_now = -m_distance(det, trk, None)
			elif metric == 'dist_2d': dist_now = -dist_ground(det, trk)              	
			elif metric == 'dist_3d': dist_now = -dist3d(det, trk)              				
			else: assert False, 'error'
			aff_matrix[d, t] = dist_now

	return aff_matrix

def greedy_matching(cost_matrix):
    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py

    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    # sort all costs and then convert to 2D
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    # assign matches one by one given the sorting, but first come first serves
    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        # if both id has not been matched yet
        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(matched_indices)

# def data_association(dets, trks, metric, threshold, algm='greedy', \
# 	trk_innovation_matrix=None, hypothesis=1, threshold_upper=float('inf'),):   
# 	"""
# 	Assigns detections to tracked object

# 	dets:  a list of Box3D object
# 	trks:  a list of Box3D object

# 	Returns 3 lists of matches, unmatched_dets and unmatched_trks, and total cost, and affinity matrix
# 	"""

# 	# if there is no item in either row/col, skip the association and return all as unmatched
# 	aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
# 	if len(trks) == 0: 
# 		return np.empty((0, 2), dtype=int), np.arange(len(dets)), [], 0, aff_matrix
# 	if len(dets) == 0: 
# 		return np.empty((0, 2), dtype=int), [], np.arange(len(trks)), 0, aff_matrix		
	
# 	# prepare inverse innovation matrix for m_dis
# 	if metric == 'm_dis':
# 		assert trk_innovation_matrix is not None, 'error'
# 		trk_inv_inn_matrices = [np.linalg.inv(m) for m in trk_innovation_matrix]
# 	else:
# 		trk_inv_inn_matrices = None

# 	# compute affinity matrix
# 	aff_matrix = compute_affinity(dets, trks, metric, trk_inv_inn_matrices)

# 	# association based on the affinity matrix
# 	if hypothesis == 1:
# 		if algm == 'hungar':
# 			row_ind, col_ind = linear_sum_assignment(-aff_matrix)      	# hougarian algorithm
# 			matched_indices = np.stack((row_ind, col_ind), axis=1)
# 		elif algm == 'greedy':
# 			matched_indices = greedy_matching(-aff_matrix) 				# greedy matching
# 		else: assert False, 'error'
# 	else:
# 		cost_list, hun_list = best_k_matching(-aff_matrix, hypothesis)

# 	# compute total cost
# 	cost = 0
# 	for row_index in range(matched_indices.shape[0]):
# 		cost -= aff_matrix[matched_indices[row_index, 0], matched_indices[row_index, 1]]

# 	# save for unmatched objects
# 	unmatched_dets = []
# 	for d, det in enumerate(dets):
# 		if (d not in matched_indices[:, 0]): unmatched_dets.append(d)
# 	unmatched_trks = []
# 	for t, trk in enumerate(trks):
# 		if (t not in matched_indices[:, 1]): unmatched_trks.append(t)

# 	# filter out matches with low affinity or too high affinity
# 	matches = []
# 	for m in matched_indices:
# 		if (aff_matrix[m[0], m[1]] < threshold) or (aff_matrix[m[0], m[1]] > threshold_upper):
# 			unmatched_dets.append(m[0])
# 			unmatched_trks.append(m[1])
# 		else:
# 			matches.append(m.reshape(1, 2))
# 	if len(matches) == 0: 
# 		matches = np.empty((0, 2),dtype=int)
# 	else: matches = np.concatenate(matches, axis=0)

# 	return matches, np.array(unmatched_dets), np.array(unmatched_trks), cost, aff_matrix

def data_association(dets, trks, metrics, thresholds, algm='greedy', 
                     trk_innovation_matrix=None, hypothesis=1, threshold_upper=float('inf')):
    """
    Assigns detections to tracked objects using multiple metrics consecutively.

    dets: a list of Box3D object (detections)
    trks: a list of Box3D object (tracks)
    metrics: list of metrics to use for association (e.g., ['iou_3d', 'dist_3d'])
    thresholds: list of thresholds for each metric
    algm: association algorithm ('greedy' or 'hungar')
    trk_innovation_matrix: required for m_dis metric
    hypothesis: 1 for single hypothesis, k for best-k matching
    threshold_upper: upper threshold to filter matches
    
    Returns matches, unmatched_dets, unmatched_trks, total cost, affinity_matrix
    """
    assert len(metrics) == len(thresholds), "Each metric must have a corresponding threshold"
    
    unmatched_dets = list(range(len(dets)))
    unmatched_trks = list(range(len(trks)))
    all_matches = []
    total_cost = 0
    combined_affinity = None

    for i, metric in enumerate(metrics):
        # Skip if there are no unmatched detections or tracks
        if not unmatched_dets or not unmatched_trks:
            break
        
        # Filter detections and tracks for current pass
        det_subset = [dets[d] for d in unmatched_dets]
        trk_subset = [trks[t] for t in unmatched_trks]
        
        # Prepare inverse innovation matrix for m_dis metric
        if metric == 'm_dis':
            assert trk_innovation_matrix is not None, 'Error: trk_innovation_matrix is required for m_dis'
            trk_inv_inn_matrices = [np.linalg.inv(trk_innovation_matrix[t]) for t in unmatched_trks]
        else:
            trk_inv_inn_matrices = None

        # Compute affinity matrix for current metric
        aff_matrix = compute_affinity(det_subset, trk_subset, metric, trk_inv_inn_matrices)
        if combined_affinity is None:
            combined_affinity = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d_idx, d in enumerate(unmatched_dets):
            for t_idx, t in enumerate(unmatched_trks):
                combined_affinity[d, t] = aff_matrix[d_idx, t_idx]

        # Perform association based on affinity matrix
        if algm == 'hungar':
            row_ind, col_ind = linear_sum_assignment(-aff_matrix)
            matched_indices = np.stack((row_ind, col_ind), axis=1)
        elif algm == 'greedy':
            matched_indices = greedy_matching(-aff_matrix)
        else:
            raise ValueError('Invalid association algorithm')

        # Filter matches by threshold
        matches = []
        for m in matched_indices:
            if aff_matrix[m[0], m[1]] < thresholds[i] or aff_matrix[m[0], m[1]] > threshold_upper:
                continue
            matches.append((unmatched_dets[m[0]], unmatched_trks[m[1]]))
            total_cost -= aff_matrix[m[0], m[1]]
        
        # Update matched and unmatched lists
        matched_dets = {m[0] for m in matches}
        matched_trks = {m[1] for m in matches}
        unmatched_dets = [d for d in unmatched_dets if d not in matched_dets]
        unmatched_trks = [t for t in unmatched_trks if t not in matched_trks]
        all_matches.extend(matches)

    # Convert matches to final format
    if len(all_matches) == 0:
        all_matches = np.empty((0, 2), dtype=int)
    else:
        all_matches = np.array(all_matches)

    return all_matches, np.array(unmatched_dets), np.array(unmatched_trks), total_cost, combined_affinity
