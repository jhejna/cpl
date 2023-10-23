# Make sure we have the conda environment set up.
CONDA_PATH=~/miniconda3/bin/activate
ENV_NAME=cpl
REPO_PATH=path/to/your/repo
USE_MUJOCO_PY=true # For using mujoco py
WANDB_API_KEY="" # If you want to use wandb, set this to your API key.

# Setup Conda
source $CONDA_PATH
conda activate $ENV_NAME
cd $REPO_PATH
unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

if $WANDB_API_KEY; then
    export WANDB_API_KEY=$WANDB_API_KEY
fi

if $USE_MUJOCO_PY; then
    echo "Using mujoco_py"
    if [ -d "/usr/lib/nvidia" ]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    fi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
fi

# First check if we have a GPU available
if nvidia-smi | grep "CUDA Version"; then
    if [ -d "/usr/local/cuda-11.8" ]; then # This is the only GPU version supported by compile.
        export PATH=/usr/local/cuda-11.8/bin:$PATH
    elif [ -d "/usr/local/cuda-11.7" ]; then # This is the only GPU version supported by compile.
        export PATH=/usr/local/cuda-11.7/bin:$PATH
    elif [ -d "/usr/local/cuda" ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        echo "Using default CUDA. Compatibility should be verified. torch.compile requires >= 11.7"
    else
        echo "Warning: Could not find a CUDA version but GPU was found."
    fi
    export MUJOCO_GL="egl"
    # Setup any GPU specific flags
else
    echo "GPU was not found, assuming CPU setup."
    export MUJOCO_GL="osmesa" # glfw doesn't support headless rendering
fi
