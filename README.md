# QCD Experiments

## Getting Started

The follow procedure has been tested on a clean installation of Ubuntu 20.04.

#### 1. Install the necessary dependencies with apt.

```bash
sudo apt install mercurial gyp ninja-build  python \
    tcpdump wireshark tshark build-essential libclang-dev \
    python3.8-dev python3.8-venv zlib1g-dev
```

#### 2. Create and activate a virtual environment

```bash
python3.8 -m venv env
source env/bin/activate
```

#### 3. Setup paths for neqo-client, etc.

TODO: Add instructions for this

