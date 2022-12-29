FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04 
# FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04
# FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04
# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel



CMD nvidia-smi
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV DEBIAN_FRONTEND=noninteractive
ENV QT_DEBUG_PLUGINS=1

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



RUN apt-get update && apt-get install -y vim python3 git python3-pip

RUN apt-get install -y libqt5gui5

# Install MMCV
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
# RUN apt-get -y install libgtk2.0-dev 
# RUN apt-get -y install libglu1

ENV CUDA_VISIBLE_DEVICES all


WORKDIR /ASL

COPY requirements.txt .


RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt



COPY . .

CMD ["python3", "app.py"]
