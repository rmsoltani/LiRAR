import numpy as np
from PIL import Image
from ..registry import PIPELINES
from collections import namedtuple
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion
import math
import numba as nb
# import matplotlib.pyplot as plt
import asyncio


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

WindowSize = namedtuple("WindowSize", ["h", "w"])

CAM_CHANS = np.array(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])

@PIPELINES.register_module
class JointBilateralExpansion:
    def __init__(self, window_size=[5,5], sigma_s=25, sigma_r=10, min_dist=1, img_size=None, confidence_threshold=0.7, cam_mask=None) -> None:
        self.window_size = WindowSize(*window_size)
        self.sigma_s=sigma_s
        self.sigma_r = sigma_r
        self.min_dist = min_dist
        self.img_size=img_size
        self.confidence_threshold=confidence_threshold

        self.max_dist = math.sqrt(self.window_size.h**2 + self.window_size.w**2)
        self.range_norm_factor = gaussian(0, self.sigma_r)
        self.spatial_norm_factor = gaussian(0, self.sigma_s)

        self.cam_mask = np.array([True] * len(CAM_CHANS), dtype=bool)

        if cam_mask and type(cam_mask) in {tuple, list, np.ndarray}:
            cam_mask = np.array(cam_mask, dtype=bool)
            if cam_mask.shape == (len(CAM_CHANS),):
                self.cam_mask = cam_mask
    
    def __call__(self, res, info):
        initial_points = res["radar"]["points"].T
        initial_times = res["radar"]["times"]

        all_new_points = np.empty((0, initial_points.shape[0]), dtype=initial_points.dtype)
        all_new_times = np.empty((0, 1), dtype=initial_times.dtype)

        for j in np.where(self.cam_mask)[0]:
        # for j in range(len(CAM_CHANS)):
            pc = RadarPointCloud(initial_points.copy())

            # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            pose_record = info["cam_poserecord"][j]
            pc.translate(-np.array(pose_record['translation']))
            pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix.T)

             # Fourth step: transform from ego into the camera.
            cs_record = info["cam_cs_record"][j]
            pc.translate(-np.array(cs_record['translation']))
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

            point_cloud = pc.points

            # point_cloud = initial_points.copy()
            # cam_translation = -np.array(info["cam_poserecord"][j]['translation']).reshape((3,1))
            # cam_rotation = Quaternion(info["cam_poserecord"][j]['rotation']).rotation_matrix.T
            # cs_translation = -np.array(info["cam_cs_record"][j]['translation']).reshape((3, 1))
            # cs_rotation = Quaternion(info["cam_cs_record"][j]['rotation']).rotation_matrix.T

            # point_cloud[:3] += cam_translation
            # point_cloud[:3, :] = cam_rotation.dot(point_cloud[:3, :])

            # point_cloud[:3] += cs_translation
            # point_cloud[:3, :] = cs_rotation.dot(point_cloud[:3, :])

            camera_intrinsic = info["ref_cam_intrinsic"][j]
            image = Image.open(info["ref_cam_path"][j])

            if self.img_size:
                camera_intrinsic[0] /= (image.width / self.img_size[0])
                camera_intrinsic[1] /= (image.height / self.img_size[1])
                image.thumbnail(self.img_size)

            new_points, new_times = self._get_expanded_points_on_cam(point_cloud, initial_times, camera_intrinsic, image)
            new_pc = RadarPointCloud(new_points.T.copy())
            new_pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            new_pc.translate(np.array(cs_record['translation']))

            new_pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
            new_pc.translate(np.array(pose_record['translation']))

            # new_points2 = new_pc.points

            all_new_points = np.vstack([all_new_points, new_points])
            all_new_times = np.vstack([all_new_times, new_times])

        # print(f"{initial_points.shape[1]} -> {all_new_points.shape[0]} (+{(all_new_points.shape[0] / initial_points.shape[1]) * 100:.2f}%)")

        res["radar"]["points"] = all_new_points
        res["radar"]["times"] = all_new_times
        res["radar"]["combined"] =  np.hstack([all_new_points, all_new_times])

        return res, info
     

    # def __call__(self, res, info):
    #     initial_points = res["radar"]["points"].T
    #     initial_times = res["radar"]["times"]

    #     num_cams = len(CAM_CHANS[self.cam_mask])

    #     point_clouds = np.empty([num_cams, *initial_points.shape])
    #     cam_translations = np.empty([num_cams, 3, 1])
    #     cam_rotations = np.empty([num_cams, 3, 3])
    #     cs_translations = np.empty([num_cams, 3, 1])
    #     cs_rotations = np.empty([num_cams, 3, 3])

    #     for i, j in enumerate(np.where(self.cam_mask)[0]):
    #         point_clouds[i, :] = initial_points
    #         cam_translations[i] = -np.array(info["cam_poserecord"][j]['translation']).reshape(cam_translations.shape[1:])
    #         cam_rotations[i] = Quaternion(info["cam_poserecord"][j]['rotation']).rotation_matrix.T
    #         cs_translations[i] = -np.array(info["cam_cs_record"][j]['translation']).reshape(cam_translations.shape[1:])
    #         cs_rotations[i] = Quaternion(info["cam_cs_record"][j]['rotation']).rotation_matrix.T

    #     point_clouds[:, :3] = point_clouds[:, :3] + cam_translations
    #     point_clouds[:, :3, :] = np.matmul(cam_rotations, point_clouds[:, :3, :])

    #     point_clouds[:, :3] = point_clouds[:, :3] + cs_translations
    #     point_clouds[:, :3, :] = np.matmul(cs_rotations, point_clouds[:, :3, :])

    #     all_new_points = np.empty((0, initial_points.shape[0]))
    #     all_new_times = np.empty((0, 1), dtype=initial_times.dtype)

    #     for i, j in enumerate(np.where(self.cam_mask)[0]):
    #         camera_intrinsic = info["ref_cam_intrinsic"][j]
    #         image = Image.open(info["ref_cam_path"][j])

    #         if self.img_size:
    #             camera_intrinsic[0] /= (image.width / self.img_size[0])
    #             camera_intrinsic[1] /= (image.height / self.img_size[1])
    #             image.thumbnail(self.img_size)

    #         new_points, new_times = self._get_expanded_points_on_cam(point_clouds[i], initial_times, camera_intrinsic, image)
    #         all_new_points = np.vstack([all_new_points, new_points])
    #         all_new_times = np.vstack([all_new_times, new_times])

    #     # print(f"{initial_points.shape[1]} -> {all_new_points.shape[0]} (+{(all_new_points.shape[0] / initial_points.shape[1]) * 100:.2f}%)")

    #     res["radar"]["points"] = all_new_points
    #     res["radar"]["times"] = all_new_times
    #     res["radar"]["combined"] =  np.hstack([all_new_points, all_new_times])

    #     return res, info
    
    
    def _get_expanded_points_on_cam(self, points, times, camera_intrinsic, image):

        depths = points[2, :]

        # coloring = depths

        viewed_points = view_points(points[:3, :], camera_intrinsic, normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > self.min_dist)
        mask = np.logical_and(mask, viewed_points[0, :] > 1)
        mask = np.logical_and(mask, viewed_points[0, :] < image.size[0] - 1)
        mask = np.logical_and(mask, viewed_points[1, :] > 1)
        mask = np.logical_and(mask, viewed_points[1, :] < image.size[1] - 1)
        viewed_points = viewed_points[:, mask]
        points = points[:, mask]
        times = times[mask]
        depths = depths[mask]
        # coloring = coloring[mask]

        rgb = np.asarray(image, dtype=np.uint8)

        all_new_points = np.empty((0, points.shape[0]))
        all_new_times = np.empty((0, 1), dtype=times.dtype)

        boxes = self._get_window_around_points(points[:3], camera_intrinsic)

        box_mask = np.ones(boxes.shape[0], dtype=bool)
        box_mask = np.logical_and(box_mask, boxes[:, 2] > 1)
        box_mask = np.logical_and(box_mask, boxes[:, 3] > 1)

        all_new_points = np.vstack([all_new_points, points.T[~box_mask]])
        all_new_times = np.vstack([all_new_times, times[~box_mask]])

        boxes = boxes[box_mask]
        points = points[:, box_mask]
        times = times[box_mask]

        x1 = np.maximum(boxes[:, 0], 0)
        y1 = np.maximum(boxes[:, 1], 0)
        x2 = np.minimum(boxes[:, 0] + boxes[:, 2], rgb.shape[1])
        y2 = np.minimum(boxes[:, 1] + boxes[:, 3], rgb.shape[0])

        wy1 = y1 - boxes[:, 1]
        wy2 = y2 - boxes[:, 1]
        wx1 = x1 - boxes[:, 0]
        wx2 = x2 - boxes[:, 0]

        # windows = np.zeros([boxes.shape[0], boxes[:, 3].max(), boxes[:, 2].max(), 3], np.uint8)


        # for i, bb in enumerate(self._get_window_around_points(points[:3], camera_intrinsic)):
        for i in range(boxes.shape[0]):
            # x, y, w, h = bb
            point = points[:, i]
            # xy, w, h = self._get_window_around(*point[:3], camera_intrinsic)

            # if h <= 1 and w <= 1:
            #     all_new_points = np.vstack([all_new_points, [point]])
            #     all_new_times = np.vstack([all_new_times, [times[i]]])
            #     continue

            # x1 = max(x, 0)
            # y1 = max(y, 0)
            # x2 = min(x + w, rgb.shape[1])
            # y2 = min(y + h, rgb.shape[0])

            # wy1 = y1 - y # 0 + (y1 - xy[1]) 
            # wy2 = y2 - y # h + (y2 - (xy[1] + h))
            # wx1 = x1 - x # 0 + (x1 - xy[0])
            # wx2 = x2 - x # w + (x2 - (xy[0] + w))

            window = np.zeros([boxes[i, 3], boxes[i, 2], 3], dtype=np.uint8)
            window[wy1[i]:wy2[i], wx1[i]:wx2[i]] = rgb[y1[i]:y2[i], x1[i]:x2[i]]

            range_kernel = self._get_range_kernel(window, slice(wy1[i], wy2[i]), slice(wx1[i], wx2[i]))
            spatial_kernel = self._get_spatial_kernel(window, slice(wy1[i], wy2[i]), slice(wx1[i], wx2[i]))

            confidence_matrix = spatial_kernel * range_kernel

            pixel_width =  self.window_size.w / window.shape[1]
            pixel_height = self.window_size.h / window.shape[0]

            valid_indexes = np.column_stack(np.where(confidence_matrix >= self.confidence_threshold))
            new_points = np.empty((valid_indexes.shape[0], *point.shape))
            new_points[:] = point
            new_points[:, 0] += pixel_width * (valid_indexes[:, 1] - window.shape[1] // 2) + pixel_width // 2
            new_points[:, 1] += pixel_height * (valid_indexes[:, 0] - window.shape[0] // 2) + pixel_height // 2

            new_times = np.full((valid_indexes.shape[0], 1), times[i], dtype=times.dtype)

            all_new_points = np.vstack([all_new_points, new_points])
            all_new_times = np.vstack([all_new_times, new_times])

        
        return all_new_points, all_new_times

    def _get_window_around(self, x, y, z, intrinsic):
        corners = np.array([
            [x - self.window_size.w // 2, x + self.window_size.w // 2],
            [y - self.window_size.h // 2, y + self.window_size.h // 2],
            [z                          , z]
        ])
        res = view_points(corners, intrinsic, normalize=True)
        return (
            [math.floor(res[0,0]), math.floor(res[1, 0])],
            max(math.floor(res[0, 1] - res[0,0]), 1),
            max(math.floor(res[1, 1] - res[1, 0]), 1)
        )
    
    def _get_window_around_points(self, points, intrinsic):
        num_points = points.shape[1]
        corners = np.empty([3, num_points * 2])
        corners[0, :num_points] = points[0, :] - (self.window_size.w // 2)
        corners[1, :num_points] = points[1, :] - (self.window_size.h // 2)
        corners[2, :num_points] = points[2, :]

        corners[0, num_points:] = points[0, :] + (self.window_size.w // 2)
        corners[1, num_points:] = points[1, :] + (self.window_size.h // 2)
        corners[2, num_points:] = points[2, :]

        res = view_points(corners, intrinsic, normalize=True)

        windows = np.empty([num_points, 4], dtype=np.int64)
        windows[:, 0] = np.floor(res[0, :num_points])
        windows[:, 1] = np.floor(res[1, :num_points])
        windows[:, 2] = np.maximum(np.floor(res[0, num_points:] - res[0, :num_points]), 1)
        windows[:, 3] = np.maximum(np.floor(res[1, num_points:] - res[1, :num_points]), 1)

        return windows

    def _get_spatial_kernel(self, window, sy: slice, sx: slice):
        spatial_kernel = np.zeros(window.shape[:2], dtype=np.float64)

        pixel_width =  self.window_size.w / window.shape[1]
        pixel_height = self.window_size.h / window.shape[0]

        p_row = window.shape[0] // 2 * pixel_height
        p_col = window.shape[1] // 2 * pixel_width

        y, x = np.ogrid[sy, sx]
        y = y * pixel_height
        x = x * pixel_width

        spatial_kernel[sy, sx] = np.sqrt((x-p_col)**2+(y-p_row)**2)
        spatial_kernel[sy, sx] = gaussian(spatial_kernel[sy, sx], self.sigma_s)
        
        # normalize
        spatial_kernel /= self.spatial_norm_factor

        return spatial_kernel
    
    def _get_range_kernel(self, window, sy, sx):
        range_kernel = np.zeros(window.shape[:2], dtype=np.float64)

        p_row = window.shape[0] // 2
        p_col = window.shape[1] // 2
        gray_window = rgb2gray(window)
        p = gray_window[p_row,p_col]

        range_kernel[sy, sx] = gaussian(np.abs(gray_window[sy, sx] - p), self.sigma_r)
        range_kernel /= self.range_norm_factor

        return range_kernel


@nb.vectorize
def gaussian(x, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power(x/sig, 2.)/2)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

