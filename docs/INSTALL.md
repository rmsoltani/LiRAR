## Installation
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/docs/INSTALL.md)'s original document.

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1 or higher
- CUDA 10.0 or higher
- CMake 3.13.2 or higher
- [APEX](https://github.com/nvidia/apex)
- [spconv](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39) 

We have tested the following versions of OS and softwares:

- OS: Ubuntu 20.04
- Python: 3.8.20
- PyTorch: 1.10.1
- spconv: 2.3.6
- CUDA: 11.3

### Basic Installation 

```bash
# basic python libraries
conda create --name centerpoint python=3.8
conda activate centerpoint
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
git clone https://github.com/tianweiy/CenterPoint.git
cd CenterPoint
pip install -r requirements.txt

# add CenterPoint to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERPOINT"
```

### Advanced Installation 

#### nuScenes dev-kit

```bash
git clone https://github.com/tianweiy/nuscenes-devkit

# add the following line to ~/.bashrc and reactivate bash (remember to change the PATH_TO_NUSCENES_DEVKIT value)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_NUSCENES_DEVKIT/python-sdk"
```

#### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-11.3/bin:$PATH
export CUDA_PATH=/usr/local/cuda-11.3
export CUDA_HOME=/usr/local/cuda-11.3
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Rotated NMS 
cd ROOT_DIR/det3d/ops/iou3d_nms
python setup.py build_ext --inplace

# Deformable Convolution (Optional and only works with old torch versions e.g. 1.1)
cd ROOT_DIR/det3d/ops/dcn
python setup.py build_ext --inplace
```

#### spconv
```bash
sudo apt-get install libboost-all-dev
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *
```
