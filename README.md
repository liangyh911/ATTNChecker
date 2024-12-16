# ATTNChecker

## Installation

There are two options to install ATTNChecker, [Docker Image](#docker-image) and [From Source](#from-source). You can choose any one of them to install ATTNChecker.

### Docker Image

You can pull and run a pre-built docker image from Docker Hub

```shell
docker pull lyh911/attnchk-pytorch:1.0
docker run --ipc=host --shm-size=512m --gpus all -it --rm lyh911/attnchk-pytorch:1.0
```

After run the docker image, you can follow the instructions in the [Usage](#usage) Section to do the tests.

### From Source

#### Our Environment

- [Anaconda](https://docs.anaconda.com/anaconda/install/) virtual environment with python 3.8.10

- [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)

- gcc 11.4.0

- Nividia A100 - 80GB

#### Create an Anaconda Environment

```shell
conda create --name attnchk python==3.8.10
conda activate attnchk
```

#### Download Source Code

```shell
git clone https://github.com/liangyh911/ATTNChecker.git
cd ATTNChecker
```

#### Install required Python Packages for ATTNChecker

```shell
pip install -r requirements.txt 
```

#### Move the Modeling Scripts from HF-moding to Huggingface Transformers Package

```shell
cp ./HF-moding/modeling_bert.py      <Path_To_Anaconda>/envs/attnchk/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py
cp ./HF-moding/modeling_gpt2.py      <Path_To_Anaconda>/envs/attnchk/lib/python3.8/site-packages/transformers/models/gpt2/modeling_gpt2.py
cp ./HF-moding/modeling_gpt_neo.py   <Path_To_Anaconda>/envs/attnchk/lib/python3.8/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py
cp ./HF-moding/modeling_roberta.py   <Path_To_Anaconda>/envs/attnchk/lib/python3.8/site-packages/transformers/models/roberta/modeling_roberta.py
```

<!-- ### Move AttnChecker Scripts to Pytorch

```shell
cp  ./OptABFT_v4/CUDABlas.cu      ./pytorch/aten/src/ATen/cuda/CUDABlas.cu
cp  ./OptABFT_v4/CUDABlas.h       ./pytorch/aten/src/ATen/cuda/CUDABlas.h
cp  ./OptABFT_v4/opt_kernels.cu   ./pytorch/aten/src/ATen/cuda/opt_kernels.cu
cp  ./OptABFT_v4/Blas.cpp         ./pytorch/aten/src/ATen/native/cuda/Blas.cpp
``` -->

#### Build Pytorch from Source

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
cd ./ATTNChecker
python ./records/cleanRecords.py
```

### ATTNChecker Running Overhead

To measure the overhead of ATTNChecker, you need to use the scripts in ```./ABFT_running_time``` folder. The default settings are

- batch size: 8

- number of training: 20

Before running the scripts, make sure to use one GPU if you have multi-GPUs on your device. You can use ```export``` command.

```shell
export CUDA_VISIBLE_DEVICES=0
```

You can also change thses settings in the scripts. Before running each test, you need to clean the previous records.

```shell
# bert
python ./records/cleanRecords.py
python ./ABFT_running_time/bertTest.py
# gpt2
python ./records/cleanRecords.py
python ./ABFT_running_time/gpt2.py
# gpt neo
python ./records/cleanRecords.py
python ./ABFT_running_time/gpt-neo.py
# roberta
python ./records/cleanRecords.py
python ./ABFT_running_time/roberta.py
```

Here is an example output of bert model. For each test, the output results may vary.

The overhead is calculated by

```math
 Overhead = {(attnchkTime-baselineTime) \over baselineTime}
```

```shell
Attention Mechanism Overhead:  0.14775651309451615
Training Overhead:  0.0552227860653921
ATTNChecker Loss:  0.5106
no ATTNChecker Loss:  0.5106
```

### Checkpoint Save and Load Overhead

Before measuring the save and load overhead of checkpoint.

You can use the scripts in ```./Checkpoint_time``` folder to test the save and load time of Checkpointing of a model. Here is an example.

```shell
# bert
python ./records/cleanRecords.py
python ./Checkpoint_time/bert.py
# gpt2
python ./records/cleanRecords.py
python ./Checkpoint_time/gpt2.py
# gpt neo
python ./records/cleanRecords.py
python ./Checkpoint_time/gpt-neo.py
# roberta
python ./records/cleanRecords.py
python ./Checkpoint_time/roberta.py
```

Here is an example output of bert model. For each test, the output results may vary. The overhead is calculated in the same way as [ATTNChecker](#attnchecker-running-overhead).

```shell
Overhead of Checkpointing:  8.522816600251735
```

### Training Loss of ATTNChecker and Baseline during 3 Epochs

You can use the scripts in ```./ABFT_epoch_loss``` folder to get the training loss during 3 epoch-training.

```shell
# bert
python ./records/cleanRecords.py
python ./ABFT_epoch_loss/bert.py
# gpt2
python ./records/cleanRecords.py
python ./ABFT_epoch_loss/gpt2.py
# gpt neo
python ./records/cleanRecords.py
python ./ABFT_epoch_loss/gpt-neo.py
# roberta
python ./records/cleanRecords.py
python ./ABFT_epoch_loss/roberta.py
```

Here is an example output of bert model. For each test, the output results may vary. The baseline is the training without ATTNChecker.

```shell
Loss of ATTNChecker: 
1st epoch:  0.5349 , 2nd epoch:  0.3071 , 3rd epoch:  0.1285
Loss of Baseline: 
1st epoch:  0.5635 , 2nd epoch:  0.3362 , 3rd epoch:  0.1312
```
