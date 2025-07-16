import pickle
from pathlib import Path
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from ..registry import PIPELINES
from pyquaternion import Quaternion

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, virtual=False):
    if virtual:
        raise NotImplementedError()
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, virtual=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), virtual=virtual).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T



def read_radar_sweep(sweep):
    min_distance = 1.0
    points_sweep = RadarPointCloud.from_file(str(sweep["radar_path"])).points
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones([points_sweep.shape[1]])

    return points_sweep.T, curr_times



def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="NuScenesDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type != "NuScenesDataset":
            raise NotImplementedError()

        nsweeps = res["lidar"]["nsweeps"]

        lidar_path = Path(info["lidar_path"])
        points = read_file(str(lidar_path), virtual=res["virtual"])

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (nsweeps - 1) == len(
            info["sweeps"]
        ), "nsweeps {} should equal to list length {}.".format(
            nsweeps, len(info["sweeps"])
        )

        for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        res["lidar"]["points"] = points
        res["lidar"]["times"] = times
        res["lidar"]["combined"] = np.hstack([points, times])

        return res, info


RADAR_RMS = {
    "RADAR_FRONT": np.array([[0, -1], [1, 0]]),
    "RADAR_FRONT_RIGHT": np.array([[1, 0], [0, 1]]),
    "RADAR_BACK_RIGHT": np.array([[0, 1], [-1, 0]]),
    "RADAR_BACK_LEFT": np.array([[0, 1], [-1, 0]]),
    "RADAR_FRONT_LEFT": np.array([[-1, 0], [0, 1]])
}
@PIPELINES.register_module
class LoadRadarPointCloudFromFile(object):
    def __init__(self, dataset="NuScenesDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.front_only = kwargs.get("front_only", False)
        self.output_global = kwargs.get("output_global", True)
        self.align_velocity = kwargs.get("align_velocity", False)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type != "NuScenesDataset":
            raise NotImplementedError()

        nsweeps = res["radar"]["nsweeps"]

        if self.front_only:
            RADAR_CHANS = ["RADAR_FRONT"]
        else:
            RADAR_CHANS = ['RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_FRONT_LEFT']

        
        point_clouds = {}
        times_by_chan = {}
        
        for chan, radar_path, timestamp in zip(RADAR_CHANS, info["radar_path"], info["radar_timestamp"]):
            point_clouds[chan] = RadarPointCloud.from_file(radar_path)
            times_by_chan[chan] = np.empty([point_clouds[chan].points.shape[1]], dtype=np.float32)
            times_by_chan[chan].fill(timestamp)


        for chan, radar_sweeps in zip(RADAR_CHANS, info["radar_sweeps"]):
            assert (nsweeps - 1) == len(radar_sweeps), f"nsweeps {nsweeps} should equal to list length {len(radar_sweeps)}."

            for sweep in radar_sweeps:
                points_sweep, times_sweep = read_radar_sweep(sweep)
                point_clouds[chan].points = np.hstack([point_clouds[chan].points, points_sweep.T])
                times_by_chan[chan] = np.concatenate([times_by_chan[chan], times_sweep], axis=0)

        points = np.empty([18, 0], dtype=np.float32)
        times = np.empty([0], dtype=np.float32)

        for p, cs_record, pose_rec, chan_times, c in zip(
                point_clouds.items(), info["radar_ref_cs_rec"], info["radar_ref_pose_rec"], times_by_chan.values(), ["b", "g", "r","c", "m"]
        ):
            chan, pc = p
            if self.output_global:
                pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
                pc.translate(np.array(cs_record['translation']))

                pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
                pc.translate(np.array(pose_rec['translation']))

            if self.align_velocity:
                pc.points[6:8, :] = RADAR_RMS[chan].dot(pc.points[6:8, :])
                pc.points[8:10, :] = RADAR_RMS[chan].dot(pc.points[8:10, :])
            
            points = np.hstack([points, pc.points])
            times = np.concatenate([times, chan_times], axis=0)


        points = points.T
        times = np.zeros([points.shape[0], 1])

        res["radar"]["points"] = points
        res["radar"]["times"] = times
        res["radar"]["combined"] = np.hstack([points, times])

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        else:
            pass 

        return res, info
