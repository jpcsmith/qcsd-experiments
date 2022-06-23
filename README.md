# QCSD: A QUIC Client-Side Website-Fingerprinting Defence Framework

This repository contains the experiment and evaluation code for the paper "QCSD: A QUIC Client-Side Website-Fingerprinting Defence Framework" (USENIX Security 2022). The Rust code for the QCSD library and test clients can be found at https://github.com/jpcsmith/neqo-qcsd.

## Software Requirements
- *Ubuntu 20.04 with bash*: All code was tested on a fresh installation of Ubuntu 20.04.
- *git, git-lfs*: Used to clone the code repository and install python packages.
- *Python 3.8 with virtual envs*: Used to create a Python 3.8 virtual environment to run the evaluation and collection scripts. Install with `sudo apt-get install python3.8 python3.8-venv python3-venv`.
- *docker >= 20.10 (sudo-less)*: Used to isolate simultaneous runs of browsers and collection scripts, as well as to allow multiple wireguard clients. The current non-root user must be able to manage containers ([install](https://docs.docker.com/engine/install/ubuntu/), [post-install](https://docs.docker.com/engine/install/linux-postinstall/)).
- *tcpdump >= 4.9.3 (sudo-less)*: Used to capture traffic traces.
- *rust (rustc, cargo) == 1.51*: Used to compile the QCSD library and test client library written in Rust.
- *Others*: Additionally, the following packages are required to build the QCSD library and test client, and can be installed with the ubuntu package manager, apt.
  ```bash
  sudo apt-get install build-essential mercurial gyp ninja-build libz-dev clang tshark texlive-xetex
  ```


## Getting Started

#### 1. Clone the repository

```bash
# Clone the repository
git clone https://github.com/jpcsmith/qcsd-experiments.git
# Change to the code directory
cd qcd-experiments
# Download resources/alexa-1m-2021-07-18.csv.gz
git lfs pull
```

#### 2. Install required Python packages

```bash
# Create and activate a virtual environment
python3.8 -m venv env
source env/bin/activate
# Ensure that pip and wheel are the latest version
python -m pip install --upgrade pip wheel
# Install the requirements using pip
python -m pip install --no-cache-dir -r requirements.txt
```

#### 3. Setup

The experiments can be run either locally or distributed across multiple machines:
- The file [ansible/distributed](ansible/distributed) contains an example of the configuration required for running the experiments distributed on multiple hosts.
- The file [ansible/local](ansible/local) contains the configuration for running the experiments locally, and is used in these instructions.

Perform the following steps:
1. Set the `gateway_ip` variable in [ansible/local](ansible/local) to the non-loopback IP address of the host, for example, the LAN IP address.
2. Change the `exp_path` variable to a path on the (local) filesystem. It can be the same path to which the repository was cloned.
3. Run the following command
   ```bash
   ansible-playbook -i ansible/local ansible/setup.yml
   ```
   - to setup the docker image for creating the web-page graphs with Chromium,
   - create, start, and test docker images for the Wireguard gateways and clients,
   - and download and build the QCSD library and test clients.

   The QCSD source code is cloned on the remote host into the third-party/ directory of the folder identified by the 'exp_path' variable in the hosts file ([ansible/local](ansible/local) or [ansible/distributed](ansible/distributed))

## Running Experiments

Ensure that the environment is setup before running the experiments.
```bash
# Activate the virtual environment if not already active
source env/bin/activate
# Set the NEQO_BIN, NEQO_BIN_MP, and LD_PATH environment vars
source env_vars
```

### Overview

The results and plots in the paper were produced using snakemake. Like GNU make, snakemake will run all dependent rules necessary to build the final target. The general syntax is

```bash
snakemake --configfile=<filename> <rulename>
```

Where `<filename>` can be [config/test.yaml](config/test.yaml) or [config/final.yaml](config/final.yaml) and `<rulename>` is the name of one of the snakemake rules found in [workflow/rules/](workflow/rules/) or the target filename. The configfile can also be set in [workflow/Snakefile](workflow/Snakefile) to avoid repeatedly specifying it on the command line.

### Mapping of Figures to Snakemake Rules

The table below details the figures and tables in the paper and the rule used to produce them. The listed output files can be found in the `results/` directory.

| Section | Figure | Rule name | Output file |
|--- |--- |--- |---
| 5. Shaping Case Studies: FRONT & Tamaraw | Figure 3 | `shaping_eval__all` |  `plots/shaping-eval-front.png`, `plots/shaping-eval-tamaraw.png`
|  | Table 2 | `overhead_eval__table` | `tables/overhead-eval.tex`
| 6.1. Defending Single Connections | Figure 4 | `ml_eval_conn__all` | `plots/ml-eval-conn-tamaraw.png`, `plots/ml-eval-conn-front.png`
| 6.2. Defending Full Web-Page Loads | Figure 5 | `ml_eval_mconn__all` | `plots/ml-eval-mconn-tamaraw.png`, `plots/ml-eval-mconn-front.png`
| | Figure 6 | `ml_eval_brows__all`| `plots/ml-eval-brows-front.png`
| E. Overhead in the Multi-connection Setting | Table 3 |  `overhead_eval_mconn__table` | `tables/overhead-eval-mconn.tex`
| F. Server Compliance with Shaping | Figure 8 | None. Instead see `workflow/notebooks/failure-analysis.ipynb` | `plots/failure-rate.png`|

## Licence

The code in this repository and associated data is released under an MIT licence as found in the LICENCE file.
