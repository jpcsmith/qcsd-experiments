# README

- **TODO**: Add a branch/tag to the neqo-qcsd repo
- **TODO**: Change to public URL of the experiments repo and merge to main
- **TODO**: Update the instructions to download the LFS file
- **TODO**: Time estimates for the rules?

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
git clone https://gitlab.inf.ethz.ch/jsmith/qcd-experiments.git
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

Where `<filename>` can be [config/test.yaml](config/test.yaml) or [config/final.yaml](config/final.yaml) and `<rulename>` is the name of one of the snakemake rules found in [workflow/rules/](workflow/rules/) or the target filename.
The table below details the figures and tables in the paper and the rule used to produce them. The listed output files can be found in the results directory.

| Section | Figure | Rule name | Output file |
|--- |--- |--- |---
| 5. Shaping Case Studies: FRONT & Tamaraw | Figure 3 | `shaping_eval__all` |  `plots/shaping-eval-front.png`, `plots/shaping-eval-tamaraw.png`
| | Table 2 | `overhead_eval__table` | `tables/overhead-eval.tex`
| 6. Defending against WF Attacks at the Client | Figure 4


### Section 5 Shaping Case Studies: FRONT & Tamaraw

```bash
snakemake --configfile=config/test.yaml shaping_eval__all
```

will create the plots
- results/plots/shaping-eval-front.png
- results/plots/shaping-eval-tamaraw.png


#### Create web-page graphs

Run
```bash
snakemake --configfile=config/small.yaml depfetch__webpage_graphs -j
```

- Version scan (5 min)
	- Scans for QUIC support in the alexa top-1m list
- Webpage graphs (10 min)
- Results are written to results/

- Set the IP address in hosts-small

cargo install .. --locked
```bash
cd third-party/
git clone https://github.com/jpcsmith/neqo-qcsd.git
cd neqo-qcsd
cargo install --path neqo-client --root ../  --locked
cargo install --path neqo-client-mp --root ../  --locked
```


```bash
export NEQO_BIN="$PWD/third-party/bin/neqo-client"
export NEQO_BIN_MP="$PWD/third-party/bin/neqo-client-mp"
```

`find third-party/neqo-qcsd/target/release/ \( -wholename '*/Release/lib/libnspr4.so' -o -wholename '*/Release/lib/libnss3.so' \) -exec cp {} third-party/lib/ \;`
