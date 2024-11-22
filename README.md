# ATTNChecker

## Installation

### Our Environment

- [Anaconda](https://docs.anaconda.com/anaconda/install/) virtual environment with python 3.8.10

- [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)

- gcc 11.4.0

- Nividia A100 - 80GB

### Create a conda environment

```shell
conda create --name attnchk python==3.8.10
conda activate attnchk
```

### Download Source Code

```shell
git clone https://github.com/liangyh911/ATTNChecker.git
cd ATTNChecker
```

### Install required Python Packages for ATTNChecker

```shell
pip install -r requirements.txt 
```

### Move the modeling scripts from HF-moding to transformers

```shell
cp ./HF-moding/modeling_bert.py      path_to_anacond/envs/attnchk/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py
cp ./HF-moding/modeling_gpt2.py      path_to_anacond/envs/attnchk/lib/python3.8/site-packages/transformers/models/gpt2/modeling_gpt2.py
cp ./HF-moding/modeling_gpt_neo.py   path_to_anacond/envs/attnchk/lib/python3.8/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py
cp ./HF-moding/modeling_roberta.py   path_to_anacond/envs/attnchk/lib/python3.8/site-packages/transformers/models/roberta/modeling_roberta.py
```

<!-- ### Move AttnChecker Scripts to Pytorch

```shell
cp  ./OptABFT_v4/CUDABlas.cu      ./pytorch/aten/src/ATen/cuda/CUDABlas.cu
cp  ./OptABFT_v4/CUDABlas.h       ./pytorch/aten/src/ATen/cuda/CUDABlas.h
cp  ./OptABFT_v4/opt_kernels.cu   ./pytorch/aten/src/ATen/cuda/opt_kernels.cu
cp  ./OptABFT_v4/Blas.cpp         ./pytorch/aten/src/ATen/native/cuda/Blas.cpp
``` -->

### Build Pytorch from Source

```shell
cd ./ATTNChecker/pytorch
conda install cmake ninja
pip install mkl-static mkl-include
pip install -r requirements.txt
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

If pytorch build successfuly, You will see in the last line of the output

```shell
Finished processing dependencies for torch==2.3.0a0+git83e4d29
```

And you also could test if cuda is available by using ```torch.cuda.is_available()```. You will get ```True``` if cuda is available.

```shell
import torch
torch.cuda.is_available()
```

## Usage

### Cleanup Records

You can use ```./records/cleanRecords.py``` to clean the existed records before running a new scripts.

```shell
python ./records/cleanRecords.py
```

### ATTNChecker Running Time

To measure the running time of ATTNChecker, you need to use the scripts in ```./ABFT_running_time``` folder. The default settings are

- batch size: 8

- test iteration: 20

Before running the scripts, make sure use one GPU if you have muti-GPUs on our device. You can ```export``` command.

```shell
export CUDA_VISIBLE_DEVICES=0
```

You can also change thses settings in the scripts. Here is an example to run gpt2.

```shell
# for gpt2
python ./ABFT_running_time/gpt2.py
```

To disable ATTNChecker, you need to replace '2' to '0' in  ```./control/AttnChecker_Mod.txt```. Then, running the script again.

### Checkpoint Save and Load Time

Before measuring the save and load time of checkpoint, please make sure you have disable ATTNChecker.

You can use the scripts in ```./Checkpoint_time``` folder to test the save and load time of Checkpointing of a model. Here is an example.

```shell
# for gpt2
python ./Checkpoint_time/gpt2.py
```
