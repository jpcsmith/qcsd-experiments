
## Getting Started

zcat, gzip, curl, split, tshark, tcpdump (with non-root capture permissions)


#### 2. Create and activate a virtual environment

```
python3.8 -m venv env
source env/bin/activate
```


#### 3. Install the Python requirements

```
# Ensure that pip and wheel are the latest version
python3 -m pip install --upgrade pip wheel

# Install the requirements using pip
python3 -m pip install --no-cache-dir -r requirements.txt
```

If the installation fails, ensure that the Python development libraries are installed and retry the above. On Ubuntu 18.04, this would be the python3.8-dev and python3-venv packages.


#### 4. Specify a configuration file

There are two configuration files, already provided: `debug.yaml` and `final.yaml`. The file `final.yaml` is the configuration used for the paper, and `debug.yaml` is a much smaller debugging configuration (fewer samples, requests, etc.).

```
# Point config.yaml to the final configuration
cd config/ && ln -s final.yaml config.yaml && cd -
```
