BootStrap: docker
From: pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Set environment variables
%environment
    export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

%post
    # To fix GPG key error when running apt-get update
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

    apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

    conda clean --all

    # Install MMCV
    pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

    # Install MMSegmentation
    git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
    cd /mmsegmentation
    export FORCE_CUDA="1"
    pip install -r requirements.txt
    pip install --no-cache-dir -e .

%runscript
    # Define what happens when the container is run
    echo "Running Singularity container with PyTorch and MMCV"
