# Docker
This is a helper project for Docker. Use it as reference.


Table of Contents:
- Installation
    - Windows
        - [Installation via WSL \& without Docker Desktop](#installation-via-wsl--without-docker-desktop)
        - [Using Docker Desktop](#installation-via-docker-desktop)
    - Linux
        - [Using apt / package manager](#installation-via-apt--package-manager)
- [Concept](#concept)
- [Why do I need this in Artificial Intelligence?](#why-do-i-need-this-in-artificial-intelligence)
- [Container Management](#container-management)
- [Image Management](#image-management)
- [Disk/Volumne Mounting](#mounting-a-disk--volume)
- [Container Lifecycle and Data](#container-lifecycle-and-data)
- [Templates](#templates)



<br><br>

---

### Installation via WSL & without Docker Desktop

First install WSL:
```cmd
wsl --install
```

Open WSL console and type:
```wsl
sudo apt update && sudo apt upgrade -y
```

```wsl
sudo apt install docker.io -y
```

```wsl
sudo service docker start
```

```wsl
sudo usermod -aG docker $USER
```

Restart your WSL now.

Verify your installation with:
```wsl
docker --version
docker run hello-world
```

Install Git in WSL:
```wsl
sudo apt install git -y
sudo apt install git-gui -y
```

```wsl
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

```wsl
git --version
```






If you need, notice following.

You can check installed WSL systems:
```cmd
wsl --list --verbose
```

And deleting with:
```cmd
wsl --unregister Ubuntu
```



Now you can code by typing (this installs and opens visual studio code):
```
code .
```



And you might want to try git gui by:
```
git gui
```


<br><br>

---

### Installation via Docker Desktop



<br><br>

---

### Installation via apt / package manager



<br><br>

---

### Concept

Docker is a platform for creating, deploying, and running applications in containers.<br>
A container packages your code, dependencies, and runtime environment together, making it portable and reproducible.<br>
It also can mount storage and therefore change/add files on your system and so it is a kind of reproducible virtual operating system. 

Images are the names of the instruction how the docker should be build. What should be installed, what base construct (os/image) should be used and which files should be copied and what name should the workspace have. These images are online in the **docker hub** available (you can also upload your own images there). For example there are images for a ubuntu container, PyTorch and TensorFlow. So you can choose an ubuntu os as start point and add your needed installations. This allows quick testing of different systems with same conditions for future reproductions.

Key points:
- Containers are lightweight virtual environments (less overhead than full VMs).
- Use Docker images as templates to spawn containers.
- Allows you to isolate dependencies, avoiding conflicts across projects.


<br><br>

---

### Why do I need this in Artificial Intelligence?

AI projects often require specific versions of frameworks (PyTorch, TensorFlow, CUDA) and system libraries. Testing and finding the right dependencies and software versions often need time and you don't want to fill your system with trash or install another os system during this process. Instead you use this virtual container solution.<br>
If you use a new architecture which need no change on your system, then you first don't need this overhead but if your work should be reproduced by others or run on another device then it might make sense to still use docker to make this process more comfortable and easier.

Docker helps you:
- Reproduce experiments easily across machines.
- Run GPU-enabled containers without altering your main system setup.
- Share setups with colleagues or for deployment.
- Avoid "it works on my machine" problems by standardizing the environment.


<br><br>

---

### Container Management

**Basic commands to run a container:**
- Pull an image from Docker Hub:
    ```bash
    docker pull ubuntu:22.04
    ```
- Start a container interactively:
    ```bash
    docker run -it --name my_container ubuntu:22.04 bash
    ```
- Start a container with GPU support (requires NVIDIA Docker runtime):
    ```bash
    docker run --gpus all -it nvidia/cuda:12.1-base bash
    ```
- List running containers:
    ```bash
    docker ps
    ```
- List running + stopped containers:
    ```bash
    docker ps -a
    ```
- Stop a container:
    ```bash
    docker stop <container_name_or_id>
    ```
- Remove a container:
    ```bash
    docker rm <container_name_or_id>
    ```
- Restart a container:
    ```bash
    docker start -ai <container_name_or_id>
    ```
- Show logs of a container:
    ```bash
    docker logs <container_name_or_id>
    ```

> On Windows you have the **Docker Desktop** running in the background


**The running command in more detail:**
```bash
docker run [OPTIONS] IMAGE [COMMAND]
```

| Flag                    | Description                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `-it`                   | Run container interactively with a terminal (`-i` keeps STDIN open, `-t` allocates a pseudo-TTY) |
| `--rm`                  | Automatically remove the container when it exits                                                 |
| `-d`                    | Run container in **detached/background mode**                                                    |
| `--name <name>`         | Assign a custom name to your container                                                           |
| `-p <host>:<container>` | Map host port to container port                                                                  |
| `-v <host>:<container>` | Mount host directory into container (persistent data)                                            |
| `--gpus all`            | Give container access to all NVIDIA GPUs                                                         |


Examples:
- Run interactively and remove container on exit:
    ```bash
    docker run -it --rm ubuntu:22.04 bash
    ```
- Run in background:
    ```bash
    docker run -d --name my_bg_container ubuntu:22.04 tail -f /dev/null
    ```
    > `tail -f /dev/null` keeps the container alive without exiting immediately.
- Run with GPU support:
    ```bash
    docker run -it --gpus all nvidia/cuda:12.1-base bash
    ```

**Accessing a Running Container**
- List running containers:
    ```bash
    docker ps
    ```
- Enter a running container (interactive shell):
    ```bash
    docker exec -it <container_name_or_id> bash
    ```
- Attach to a running container (direct terminal, less flexible):
    ```bash
    docker attach <container_name_or_id>
    ```


**Building Docker Images**
Basic construct:
```bash
docker build [OPTIONS] PATH
```
- `PATH` → directory containing your Dockerfile (for example `.`)
- `-t <name>:<tag>` → assign name and optional tag (latest by default)
- `-f <Dockerfile>` → use a specific Dockerfile (default is Dockerfile)
- `--no-cache` → rebuild without using cache (useful if dependencies changed)

Examples:
```bash
docker build -t my_ai_image:1.0 .
docker build -f Dockerfile.dev -t my_dev_image .
docker build --no-cache -t my_fresh_image .
```

- Build with default name (latest tag):
    ```bash
    docker build -t my_image_name .
    ```
- Build with custom name and tag:
    ```bash
    docker build -t my_custom_name:1.0 .
    ```
- Build using a different Dockerfile:
    ```bash
    docker build -f Dockerfile.dev -t my_dev_image .
    ```
    ```bash
    docker build -f my_ptorch_project.Dockerfile -t my_dev_image .
    ```
    > `-f` points to the Dockerfile you want to use (default is Dockerfile).



<br><br>

---

### Image Management

Create a Dockerfile to define a **custom image**. Comments are `#`:
```dockerfile
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "main.py"]
```

Or:
```dockerfile
# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set default command
CMD ["python", "train.py"]
```

**Key instructions are:**

| Command                                | Description                                                             |
| -------------------------------------- | ----------------------------------------------------------------------- |
| `FROM <image>`                         | Base image to start from (e.g., `python:3.11`, `nvidia/cuda:12.1-base`) |
| `RUN <command>`                        | Execute command in build stage (e.g., install packages)                 |
| `WORKDIR <path>`                       | Set working directory for subsequent commands                           |
| `COPY <src> <dest>`                    | Copy files/folders from host into image                                 |
| `ADD <src> <dest>`                     | Like COPY but can handle URLs and auto-extract archives                 |
| `ENV <key>=<value>`                    | Set environment variables inside the container                          |
| `EXPOSE <port>`                        | Document ports that container will listen on                            |
| `CMD ["executable","param1","param2"]` | Default command when container starts                                   |
| `ENTRYPOINT ["executable","param1"]`   | Like CMD but fixed; CMD arguments are appended                          |
| `USER <username>`                      | Switch user for running commands or container                           |
| `VOLUME <path>`                        | Define mount points for external storage                                |
| `ARG <name>`                           | Build-time argument, used in `RUN` or other instructions                |


<br><br>

**Image Usage**<br>

After you might want to build + run the image or upload your image to the docker hub (this is partwise described in more detail in [the docker container management section](#container-management)).
- Build the image:
    ```bash
    docker build -t my_ai_image:latest .
    ```
- Run the image:
    ```bash
    docker run -it --name my_ai_container my_ai_image:latest
    ```
- Tag and push to Docker Hub:
    ```bash
    docker tag my_ai_image username/my_ai_image:latest
    docker push username/my_ai_image:latest
    ```


**Basic commands for docker images:**
- List images:
    ```bash
    docker images
    ```
- Remove image:
    ```bash
    docker rmi <image_name_or_id>
    ```
- Inspect image (see layers, size, metadata):
    ```bash
    docker inspect <image_name_or_id>
    ```
- Tag an existing image (give it another name/tag):
    ```bash
    docker tag my_ai_image:latest my_registry/my_ai_image:1.0
    ```
- Push to a registry (like Docker Hub):
    ```bash
    docker push my_registry/my_ai_image:1.0
    ```


**Important Things to know:**
1. **Layering** – Each RUN, COPY, etc. creates a new layer. Combine commands where possible to reduce image size.
2. **Cache** – Docker caches layers for faster rebuilds. Use --no-cache if you want a clean build.
3. **Lightweight Base Images** – Use slim images (python:3.11-slim) or specialized AI images (nvidia/cuda) to reduce size and dependency conflicts.
4. **Environment & Dependencies** – Always pin versions (tensorflow==2.14, torch==2.3) for reproducibility.
5. **Default Command** – Use CMD or ENTRYPOINT wisely so container behaves as expected when run.
6. **Volumes for Data** – Don’t bake datasets into the image; mount directories at runtime for large datasets.
7. **Sudo** – Don’t use `sudo` inside a Dockerfile, you are the root user by default and it can result in errors when still using such commands.


<br><br>

---

### Mounting a Disk / Volume

You have **two main options** to make data persistent or accessible outside a container:

#### **A. At runtime (`docker run`)**

You can mount a directory from your host machine into the container:

```bash
docker run -v /path/on/host:/path/in/container my_image
```

* `/path/on/host` → folder on your computer
* `/path/in/container` → folder inside the container
* Any changes in the container to this path **will persist on the host**, even if the container is stopped or removed.

Example for AI training:

```bash
docker run -v ~/ai_models:/workspace/models my_image
```

* Trains AI inside `/workspace/models` in the container
* Saves the models directly to `~/ai_models` on your PC

You can also mount multiple volumes with multiple `-v` flags.

<br>

#### **B. In the Dockerfile (during build)**

You can define **volumes** in the image:

```dockerfile
VOLUME /workspace/data
```

* This tells Docker: “This directory will store persistent data.”
* When a container is created from this image, Docker will automatically create an **anonymous volume**.
* But **you won’t control the path on the host unless you mount it at runtime**. So for AI models, runtime mounting is usually better.


<br><br>

---

### Container lifecycle and data

Container can storage data or directly use and save data to the host (via a mounted dis/volumne). It is recommended to use the host as storage and not the container then in most cases this makes everything only more complicated, because the data have to pass afterwards to the host and after multiple starts this can consume more storage, but in some cases this still can make sense. This storage and stopping and removing is covered here in more detail:

* **Stopping a container**:
  ```bash
  docker stop <container_name>
  ```
  * Container stops but **all changes inside the container remain**.
* **Accessing a stopped container**:
  ```bash
  docker start -ai <container_name>
  ```
    * `-a` attaches to STDOUT/STDERR + `-i` keeps STDIN open
    * You can also **copy data from a stopped container** to your host:
        ```bash
        docker cp <container_name>:/path/in/container /path/on/host
        ```
        Example:
        ```bash
        docker cp my_container:/workspace/models ./models_backup
        ```
* **Removing a container**
    - `docker rm <container_name>` → deletes the container **permanently**, including all data **not stored in a mounted volume**.
    - If you mounted a host folder (`-v /host:/container`), that data **stays on your host**.

<br>

**Best practice for AI workflows**

1. Always **mount a host folder** for your models and datasets.
2. Use Docker volumes if you want **automatic persistence** but don’t need host access.
3. Containers are disposable—think of them as ephemeral **environments**, not storage.

<br><br>

<!--
We already talked about the end of a Container but the Container lifecycle of course also need a beginning. And the running commands in a docker can have 3 different versions.
1. Running command defined in the docker image
2. Running command given at the docker run command
3. Running command is given interactively
-->

**Determine What to Run in Docker**<br>

A Docker container has a lifecycle that **starts** when you create and run it, continues while it’s running, and ends when you stop or remove it (what already got covered above). Understanding this is crucial, especially for AI projects where long-running training scripts or services are common.

You start a container with:
```bash
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
```
- `IMAGE` – the Docker image you want to run
- `COMMAND` – optional, overrides the default command in the image
- `[ARG...]` – optional arguments for the command

When you run a container, there are three ways commands can be executed:
1. **Command defined in the Docker image (CMD / ENTRYPOINT)**
    - Each Dockerfile usually defines a default command using CMD or ENTRYPOINT.
    - If you don’t specify a command in docker run, this command will execute automatically.

    Example:
    ```dockerfile
    # Dockerfile
    FROM pytorch/pytorch:2.3.1-cuda11.8
    WORKDIR /workspace
    COPY train.py /workspace/train.py
    CMD ["python", "train.py"]
    ```
    Run:
    ```bash
    docker run --gpus all pytorch-training
    ```
    - This runs python train.py automatically.
    - Ideal for automated scripts or services.
2. **Command given at `docker run`**
    - You can override the default CMD at runtime by providing a command at the end of docker run.
    
    Example:
    ```bash
    docker run --gpus all pytorch-training python eval.py
    ```
    - Overrides the default CMD (train.py) and runs eval.py instead.
    - Useful when you want to run multiple scripts without creating multiple images.
3. **Interactive / manual command inside a running container**
    - You can start a container interactively to execute commands manually.
    - Use -it (interactive + terminal) to open a shell inside the container.
    
    Example:
    ```bash
    docker run -it --gpus all pytorch-training /bin/bash
    ```
    Inside of the container:
    ```bash
    python train.py
    python eval.py
    ls /workspace
    ```
    - Gives full control over the container.
    - Useful for debugging or exploring the environment.


<br><br>

---

### Templates

**Most important images:**
* **NVIDIA / CUDA + Deep Learning**
    * `nvcr.io/nvidia/pytorch:<version>-py3` – PyTorch with CUDA (GPU)
    * `nvcr.io/nvidia/tensorflow:<version>-py3` – TensorFlow with CUDA (GPU)
    * `nvidia/cuda:<version>-cudnn<version>-devel-ubuntu<version>` – CUDA + cuDNN development (GPU)
    * `nvidia/cuda:<version>-runtime-ubuntu<version>` – Minimal CUDA runtime (GPU)
* **PyTorch Official**
    * `pytorch/pytorch:<version>-cpu` – CPU-only
    * `pytorch/pytorch:<version>-cuda<version>` – GPU-enabled
* **TensorFlow Official**
    * `tensorflow/tensorflow:<version>` – CPU-only
    * `tensorflow/tensorflow:<version>-gpu` – GPU-enabled
* **Jupyter / Notebook Images**
    * `jupyter/base-notebook` – CPU-only, minimal environment
    * `jupyter/datascience-notebook` – CPU-only, includes Python/R & libraries
    * `jupyter/tensorflow-notebook` – GPU-enabled if based on TensorFlow GPU image
    * `jupyter/pytorch-notebook` – GPU-enabled if based on PyTorch GPU image
* **Other Useful Images**
    * `conda/miniconda3` – CPU-only, can install GPU libraries manually
    * `python:<version>` – CPU-only, for fully custom setups


<br><br>


**Here are a few common templates for AI projects:**

1. PyTorch + CUDA
    ```dockerfile
    FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
    RUN apt update && apt install -y python3-pip git
    RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    WORKDIR /workspace
    CMD ["/bin/bash"]
    ```
2. Minimal PyTorch for Inference
    ```dockerfile
    FROM python:3.11-slim
    RUN pip install torch torchvision torchaudio
    WORKDIR /workspace
    COPY ./models /workspace/models
    COPY ./inference.py /workspace
    CMD ["python", "inference.py"]
    ```
3. TensorFlow + GPU
    ```dockerfile
    FROM tensorflow/tensorflow:2.15.0-gpu
    WORKDIR /workspace
    COPY . /workspace
    RUN pip install -r requirements.txt
    CMD ["python", "train.py"]
    ```
4. Jupyter Notebook for AI Experiments
    ```dockerfile
    FROM jupyter/tensorflow-notebook:latest
    WORKDIR /workspace
    COPY . /workspace
    CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
    ```
5. AI Training with Mounted Dataset
    ```dockerfile
    FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
    RUN apt update && apt install -y python3-pip git
    RUN pip install torch torchvision torchaudio
    WORKDIR /workspace
    VOLUME /workspace/data
    CMD ["/bin/bash"]
    ```
    Example run:
    ```bash
    docker run -v ~/datasets:/workspace/data -it my_pytorch_image
    ```
6. Multi-GPU Training
    ```dockerfile
    FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
    RUN apt update && apt install -y python3-pip git
    RUN pip install torch torchvision torchaudio
    WORKDIR /workspace
    COPY . /workspace
    CMD ["python", "-m", "torch.distributed.launch", "--nproc_per_node=4", "train.py"]
    ```
7. AI Experiment with MLflow Tracking
    ```dockerfile
    FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
    RUN apt update && apt install -y python3-pip git
    RUN pip install torch torchvision torchaudio mlflow
    WORKDIR /workspace
    COPY . /workspace
    CMD ["mlflow", "run", "."]
    ```
8. Simple PyTorch and with CUDA
    ```dockerfile
    # Use follwing command for building
    #     docker build -t pytorch-cuda-test -f pytorch_cuda_check.Dockerfile .
    # And running with:
    #     docker run --rm --gpus all pytorch-cuda-test
    #     docker run --runtime nvidia --rm --gpus all pytorch-cuda-test
    #     docker run --runtime=nvidia --rm --gpus all pytorch-cuda-test
    # 

    # Minimal PyTorch + CUDA Test Image
    FROM nvcr.io/nvidia/pytorch:25.04-py3
    # FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
    # FROM nvidia/cuda:12.0.1-devel-ubuntu20.04
    # FROM nvidia/cuda:12.0.1-devel-ubuntu22.04
    # FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
    # FROM nvcr.io/nvidia/pytorch:23.09-py3 

    # Verhindert interaktive Prompts
    ENV DEBIAN_FRONTEND=noninteractive

    # Python + Pip installieren
    # RUN apt-get update && apt-get install -y --no-install-recommends \
    #     python3 \
    #     python3-pip \
    #     && rm -rf /var/lib/apt/lists/*

    # Check CUDA installations
    RUN ls -l /usr/local/cuda*

    # PyTorch mit passender CUDA-Version installieren
    # RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1 \
    #     --index-url https://download.pytorch.org/whl/cu121
    # RUN pip install torch==2.3.1+cu120 torchvision==0.18.1+cu120 torchaudio==2.3.1 \
    #     --index-url https://download.pytorch.org/whl/cu120
    # RUN pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1 \
    #     --index-url https://download.pytorch.org/whl/cu118

    # Testscript hinzufügen
    COPY test_cuda.py /workspace/test_cuda.py

    WORKDIR /workspace

    CMD ["python3", "test_cuda.py"]
    ```


