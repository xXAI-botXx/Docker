# Docker
This is a helper project for Docker using in machine learning

Table of Contents:
- [Docker](#docker)
    - [Installation via WSL \& without Docker Desktop](#installation-via-wsl--without-docker-desktop)






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









