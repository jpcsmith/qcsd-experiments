# Create an image for fetching websites with Neqo and Wireguard
# 
# A wireguard configuration needs to be mounted to 
# /etc/wireguard/wg0.conf to run the container
#
# =================
# Install Wireguard
# =================
FROM python:3.9-slim-buster

RUN echo "deb http://deb.debian.org/debian buster-backports main" \
        > /etc/apt/sources.list.d/backports.list \
    && apt-get -qq update \
    && apt-mark hold nftables \
    && apt-get -qq install -y iptables iproute2 wireguard procps wget iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# =================
# Copy in configuration files
# =================
COPY *.conf /etc/wireguard/
COPY connect /usr/local/sbin/connect

# Path to search for shared objects for NEQO, will need to be mounted
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/neqo/"

ENTRYPOINT ["connect"]
