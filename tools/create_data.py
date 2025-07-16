from pathlib import Path

import fire

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database

def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True, virtual=False, modalities=["lidar"], base_suffix="", create_infos=True):
    if create_infos:
        nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero, modalities=modalities, base_suffix=base_suffix)
    if version == 'v1.0-trainval' or version == "v1.0-mini":
        if type(nsweeps) in (list, tuple):
            num_lidar_sweeps, num_radar_sweeps, *_ = nsweeps
            sweep_str = f"{num_lidar_sweeps:02d}_{num_radar_sweeps:02d}"
        else:
            num_lidar_sweeps = nsweeps
            num_radar_sweeps = nsweeps
            sweep_str = f"{nsweeps:02d}"


        suffix = base_suffix
        suffix += "velo" if "lidar" in modalities else ""
        suffix += "_radar" if "radar" in modalities else ""

        create_groundtruth_database(
            "NUSC",
            root_path,
            Path(root_path) / f"infos_train_{sweep_str}sweeps_with{suffix}_filter_{filter_zero}.pkl",
            nsweeps=num_lidar_sweeps,
            nrsweeps=num_radar_sweeps,
            virtual=virtual,
            modalities=modalities,
            base_suffix=base_suffix
        )
    

if __name__ == "__main__":
    fire.Fire()
