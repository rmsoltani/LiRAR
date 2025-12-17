import numpy as np
from scipy.spatial import KDTree
from ..registry import PIPELINES
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import PointCloud, LidarPointCloud

@PIPELINES.register_module
class LidarPlusRadarFusion(object):
    def __init__(
        self, 
        radar_feature_mask=None,
        max_fusion_radius=None, 
        fusion_workers=-1, 
        filter_unique_radar=False, 
        fuse_on_global=True, 
        append_radar=True, 
        confidence_config=None,
        ndims=3,
    ) -> None:
        """
        :param radar_feature_mask: Either a list/tuple contain the indexes of the used features or a boolean mask.
        If excluded, no features will be used. (x,y,z coors do are not counted as 'features')
        The features used must align with those used when creating the gt_database
        
        Feature indexes:
        0        1  2   3  4  5       6       7                8           9    10     11            12   13     14
        dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        """

        self.radar_feature_mask = np.array([False, False, False], dtype=bool)
        self.max_fusion_radius = max_fusion_radius
        self.fusion_workers = fusion_workers
        self.filter_unique_radar = filter_unique_radar
        self.fuse_on_global = fuse_on_global
        self.append_radar = append_radar
        self.confidence_config = confidence_config
        self.ndims = ndims

        print("Filter unique:", self.filter_unique_radar)
        print("Confidence Config:", self.confidence_config)

        if radar_feature_mask is not None:
            radar_feature_mask = np.array(radar_feature_mask)

            if radar_feature_mask.dtype == bool: 
                assert len(radar_feature_mask == 15), "Bool feature mask must map to all 15 features"
                self.radar_feature_mask = np.concatenate([
                    self.radar_feature_mask, 
                    np.array(radar_feature_mask, dtype=bool)
                ])
            elif radar_feature_mask.dtype == int:
                self.radar_feature_mask = np.concatenate([
                    self.radar_feature_mask,
                    np.array([i in radar_feature_mask for i in range(15)], dtype=bool),
                ])
            else:
                raise ValueError("Invalid radar_feature_mask")
            

    def __call__(self, res, info):
        if self.fuse_on_global:
            lidar_points = self._to_global_lidar_points(np.copy(res["lidar"]["points"]), info)
        else:
            lidar_points = res["lidar"]["points"]
        
        lidar_times = res["lidar"]["times"]
        radar_points = res["radar"]["points"]
        radar_times = res["radar"]["times"]

        if self.filter_unique_radar:
            radar_points, radar_times = self._get_unique_radar(radar_points, radar_times)

        fused_points = self._get_fused_points(lidar_points, radar_points)
        if self.append_radar:
            fused_times = np.concatenate([lidar_times, radar_times], dtype=np.float32)
        else:
            fused_times = lidar_times

        if self.fuse_on_global:
            fused_points = self._from_global_lidar_points(fused_points, info).astype(np.float32)

        else:
            fused_points = fused_points.astype(np.float32)

        res["lidar"]["points"] = fused_points
        res["lidar"]["times"] = fused_times
        res["lidar"]["combined"] = np.hstack([fused_points, fused_times])

        return res, info

    def _get_fused_points(self, lidar_points, radar_points):

        radar_coords = radar_points[:, :self.ndims]
        lidar_coords = lidar_points[:, :self.ndims]

        radar_features = radar_points[:, self.radar_feature_mask]

        extended_lidar_points = np.hstack((lidar_points, np.zeros([lidar_points.shape[0], radar_features.shape[1]])))
        extended_radar_points = np.hstack((radar_points[:, :3], np.zeros([radar_points.shape[0], 1]), radar_features))

        if self.confidence_config != None:
            base_conf = self.confidence_config.get("base", 0)
            extended_lidar_points = np.hstack([extended_lidar_points, np.full([extended_lidar_points.shape[0], 1], base_conf)])
            extended_radar_points = np.hstack([extended_radar_points, np.full([extended_radar_points.shape[0], 1], base_conf)])

        for curr, closest in self._get_closest_index(lidar_coords, radar_coords):
            extended_lidar_points[curr, lidar_points.shape[1]:lidar_points.shape[1]+radar_features.shape[1]] = radar_features[closest]
            if self.confidence_config != None:
                extended_lidar_points[curr, -1] = self._get_confidence_score(lidar_points[curr], radar_points[closest], config=self.confidence_config)
        
        if self.append_radar:
            fused_points = np.concatenate([extended_lidar_points, extended_radar_points])
        else:
            fused_points = extended_lidar_points

        return fused_points
    
    def _get_closest_index(self, lidar_coords, radar_coords):
        if self.max_fusion_radius is None:
            return

        tree = KDTree(radar_coords)
        curr = 0
        _, indecies = tree.query(lidar_coords, k=1, distance_upper_bound=self.max_fusion_radius, workers=self.fusion_workers)
        for i in indecies:
            if i != tree.n:
                yield curr, i
            curr += 1
    
    def _get_confidence_score(self, lidar_point, radar_point, config={"dist": "exp"}):
        dist_ndims = config.get("dist_ndims", 2) # number dimensions to calculate distance in
        dist_formula = config.get("dist") # formula to use to calculate distance confidence
        dist_exp_pow = config.get("dist_exp_pow", 5) # the power to in exp formula use when dist=exp

        rcs_min = config.get("rcs_min", -5.0)
        rcs_max = config.get("rcs_max", 50.0)
        rcs_formula = config.get("rcs") # formula to use to calculate rcs confidence
        rcs_exp_pow = config.get("rcs_exp_pow", 5) # the power to in exp formula use when rcs=exp
        
        merge_formula = config.get("merge_formula") # formula to use to merge distance and rcs confidence scores

        combine_pow = config.get("combine_pow", 3) # the power to use when method=combine

        dist = np.linalg.norm(lidar_point[:dist_ndims] - radar_point[:dist_ndims])
        norm_rcs = (np.clip(radar_point[3], rcs_min, rcs_max) - rcs_min) / (rcs_max - rcs_min)

        if dist_formula == "sigmoid":
            conf_dist = 0.5 - np.tanh((4 * dist / self.max_fusion_radius) - 2) / 2
        elif dist_formula == "exp":
            conf_dist = 0.5 - ((2 * dist / self.max_fusion_radius - 1) ** dist_exp_pow) / 2
        elif dist_formula is not None:
            raise ValueError(f"Invalid dist mode: {dist_formula}")
        
        if rcs_formula == "exp":
            conf_rcs = ((((2 * norm_rcs) - 1) ** rcs_exp_pow) / 2) + 0.5
        elif rcs_formula is not None:
            raise ValueError(f"Invalid rcs mode: {rcs_formula}")
        
        if dist_formula and rcs_formula:
            if merge_formula == "combine":
                score = _combine_scores(conf_dist, conf_rcs, combine_pow)
                return score
            elif merge_formula == "conflate":
                return (conf_dist * conf_rcs) / ((conf_dist * conf_rcs) + ((1 - conf_dist) * (1 - conf_rcs)))
            raise ValueError(f"Invalid merge_formula: {merge_formula}")
        elif dist_formula:
            return conf_dist
        elif rcs_formula:
            return conf_rcs
        
        raise ValueError(f"Invalid confidence config: {config}")


    def _get_unique_radar(self, radar_points, radar_times):
        points, indexes =  np.unique(radar_points, return_index=True, axis=0)
        return points, radar_times[indexes]

    def _to_global_lidar_points(self, lidar_points, info):
        lidar_pc = LidarPointCloud(lidar_points.T)
        lidar_pc.rotate(Quaternion(info["ref_cs_rec"]['rotation']).rotation_matrix)
        lidar_pc.translate(np.array(info["ref_cs_rec"]['translation']))

        # Second step: transform from ego to the global frame.
        lidar_pc.rotate(Quaternion(info["ref_pose_rec"]['rotation']).rotation_matrix)
        lidar_pc.translate(np.array(info["ref_pose_rec"]['translation']))

        return lidar_pc.points.T
    
    def _from_global_lidar_points(self, lidar_points, info):
        if self.confidence_config:
            lidar_pc = EnhancedLiRARPointCloud(lidar_points.T)
        else:
            lidar_pc = BasicLiRARPointCloud(lidar_points.T)

        lidar_pc.translate(-np.array(info["ref_pose_rec"]['translation']))
        lidar_pc.rotate(Quaternion(np.array(info["ref_pose_rec"]['rotation'])).rotation_matrix.T)

        lidar_pc.translate(-np.array(info["ref_cs_rec"]['translation']))
        lidar_pc.rotate(Quaternion(np.array(info["ref_cs_rec"]['rotation'])).rotation_matrix.T)

        return lidar_pc.points.T


class BasicLiRARPointCloud(PointCloud):
    def nbr_dims(self):
        return 7
    
    def from_file(self):
        raise NotImplementedError()
    
class EnhancedLiRARPointCloud(PointCloud):
    def nbr_dims(self):
        return 8
    
    def from_file(self):
        raise NotImplementedError()

def _combine_scores(x, y, p):
    s = x + y - 1
    if s >= 0:
        return (s ** p) / 2 + 0.5
    return 0.5 - (abs(s) ** p) / 2