import pickle
from pathlib import Path
import os 
import numpy as np

from det3d.core import box_np_ops
from det3d.datasets.dataset_factory import get_dataset
from tqdm import tqdm

dataset_name_map = {
    "NUSC": "NuScenesDataset",
}


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_path=None,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    virtual=False,
    modalities=["lidar"],
    base_suffix="",
    **kwargs,
):
    pipeline = [
        {
            "type": "LoadPointCloudFromFile",
            "dataset": dataset_name_map[dataset_class_name],
        },
        {"type": "LoadPointCloudAnnotations", "with_bbox": True},
    ]
    if "radar" in modalities:
        pipeline += [
            {
                "type": "LoadRadarPointCloudFromFile",
                "dataset": dataset_name_map[dataset_class_name],
                "align_velocity": True
            },
            # {
            #     "type": "JointBilateralExpansion",
            #     "window_size": [2.36,2.36],
            #     "min_dist": 1,
            #     "img_size": [800,450],
            #     "confidence_threshold": 0.075,
            #     "sigma_r": 12,
            #     "sigma_s": 0.5,
            # },
            {
                "type": "LidarPlusRadarFusion", 
                "radar_feature_mask": [2,5,6], 
                "filter_unique_radar": True,
                "max_fusion_radius": 1,
                "append_radar": True,
                "ndims": 3,
                "confidence_config": {
                    "base": 0.5,
                    "dist": "exp",
                    "dist_exp_pow": 5,
                    "rcs": "exp",
                    "rcs_exp_pow": 3,
                    "merge_formula": "combine",
                    "combine_pow": 3
                }
            }
        ]

    if "nsweeps" in kwargs:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path,
            root_path=data_path,
            pipeline=pipeline,
            test_mode=True,
            nsweeps=kwargs["nsweeps"],
            nrsweeps=kwargs.get("nrsweeps", kwargs["nsweeps"]),
            virtual=virtual
        )
        nsweeps = dataset.nsweeps
        nrsweeps = dataset.nrsweeps
    else:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline
        )
        nsweeps = 1
        nrsweeps = 1

    root_path = Path(data_path)

    if dataset_class_name == "NUSC": 
        suffix = base_suffix
        suffix += "velo" if "lidar" in modalities else ""
        suffix += "_radar" if "radar" in modalities else ""
        if db_path is None:
            if virtual:
                db_path = root_path / f"gt_database_{nsweeps}_{nrsweeps}sweeps_with{suffix}_virtual"
            else:
                db_path = root_path / f"gt_database_{nsweeps}_{nrsweeps}sweeps_with{suffix}"
        if dbinfo_path is None:
            if virtual:
                dbinfo_path = root_path / f"dbinfos_train_{nsweeps}_{nrsweeps}sweeps_with{suffix}_virtual.pkl"
            else:
                dbinfo_path = root_path / f"dbinfos_train_{nsweeps}_{nrsweeps}sweeps_with{suffix}.pkl"
    else:
        raise NotImplementedError()

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    for index in tqdm(range(len(dataset))):
        image_idx = index
        # modified to nuscenes
        sensor_data = dataset.get_sensor_data(index)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        if nsweeps > 1: 
            points = sensor_data["lidar"]["combined"]
        else:
            points = sensor_data["lidar"]["points"]
            
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]

        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue 
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)

                filepath = os.path.join(str(db_path), names[i], filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    try:
                        gt_points.tofile(f)
                    except:
                        print("process {} files".format(index))
                        break

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = os.path.join(db_path.stem, names[i], filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)
