# Radar Fusion Into Lidar-Based Detection

> [**Radar Fusion Into Lidar Based Detection**](),            
> Reza Soltani, John Lee, Nathaniel Sigafoos,        


<!--
    @article{soltani2026lirar,
      title={Radar Fusion Into Lidar-Based Detection},
      author={Soltani, Reza and Lee, John, Sigafoos, Nathaniel},
      journal={MDPI},
      year={2026},
    }
-->

## Abstract
Perception is a fundamental component of autonomous driving systems. It involves the detection and tracking of objects surrounding an autonomous vehicle (AV) using multiple sensors mounted on the vehicle. Fusing these sensors to create multimodal data improves metric accuracy and improves system robustness. This results in a more sophisticated and reliable perception system. Many studies have been performed on the topics of radar-camera fusion and LiDAR-camera fusion. However, by comparison, far fewer have investigated radar-LiDAR fusion. Existing radar-LiDAR fusion approaches are often presented as complicated and difficult to implement properly. In this paper, we propose LiRAR, an early-fusion approach that allows for the near-seamless integration of radar into models designed around LiDAR. We evaluate the effectiveness of this approach on the nuScenes dataset, where we observe an increase of up to 1.57\% in mAP across all classes, achieving a score of 60.14. In addition, we analyze the strengths and limitations of the approach and discuss potential directions for further development. The code is available at https://github.com/rmsoltani/LiRAR.


## Main results


#### 3D detection on nuScenes val set 

|         |  MAP ↑  | NDS ↑ |
|---------|---------|-------|
|  LiRAR  |  60.14  | 67.56 |   
   

All results are tested on a Nvidia A100 GPU with batch size 4.


## Usage

### Installation

Please refer to [INSTALL](docs/INSTALL.md) to set up libraries needed for distributed training and sparse convolution.

### Benchmark Evaluation and Training 

Please refer to [NUSC](docs/NUSC.md) to prepare the data. Then follow the instruction there to reproduce our detection results. All detection configurations are included in [configs](configs).


## License

LiRAR is release under MIT license (see [LICENSE](LICENSE)). It is developed based on a forked version of [CenterPoint](https://github.com/tianweiy/CenterPoint). See the [NOTICE](docs/NOTICE) for details. Note that the nuScenes dataset is under non-commercial licenses.