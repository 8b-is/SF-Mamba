ARG PYTORCH="2.7.0"
ARG CUDA="12.6"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 ffmpeg libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

#
# base
#
SHELL ["/bin/bash", "-cu"]

RUN pip3 install --upgrade pip 'pip<25.3'


# Install requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir

RUN pip3 install blinker --no-cache-dir &&\
    pip install opencv-fixer==0.2.5 --no-cache-dir &&\
    python -c "from opencv_fixer import AutoFix; AutoFix()"

# regarding MMLab
RUN pip install -U openmim --no-cache-dir
RUN mim install mmengine --no-cache-dir
RUN mim install "mmcv>=2.0.0,<2.2.0" --no-cache-dir
RUN mim install "mmdet>=3.3.0" --no-cache-dir
RUN mim install mmpretrain --no-cache-dir &&\
    mim install "mmsegmentation>=1.0.0" --no-cache-dir &&\
    pip uninstall xtcocotools -y &&\
    pip install --no-binary xtcocotools xtcocotools # xtcocotools for avoiding numpy version conflict 


# fix mmpretrain to cope with bad transformers's version dependency
COPY mmpretrain_fix_patch.py mmpretrain_fix_patch.py
RUN python mmpretrain_fix_patch.py
