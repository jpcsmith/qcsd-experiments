# Create an image for fetching TCP logs for web-pages

# ================
# Install Chromium
# ================
FROM python:3.7-slim-buster AS chromium

# Install the dependencies
RUN apt-get -qq update && apt-get -qq install -y \
            unzip gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 \
            libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 ethtool \
            libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 \
            libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 \
            libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 \
            libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates libgbm1 \
            fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget \
        && rm -rf /var/lib/apt/lists/*

ARG chromium_commit=870763

# Download and unzip chromium
RUN wget -q "https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o/Linux_x64%2F${chromium_commit}%2Fchrome-linux.zip?alt=media" \
            -O chrome-linux.zip \
        && unzip -q chrome-linux.zip \
        && rm chrome-linux.zip \
        && ln -s /chrome-linux/chrome /usr/local/bin/chrome

# Download and unzip the driver
RUN wget -q "https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o/Linux_x64%2F${chromium_commit}%2Fchromedriver_linux64.zip?alt=media" \
            -O chromedriver.zip \
        && unzip -q chromedriver.zip \
        && mv chromedriver_linux64/chromedriver /usr/local/bin \
        && rm chromedriver.zip \
        && rmdir chromedriver_linux64

# =======================================
# Install the script and its dependencies
# =======================================
FROM chromium

RUN apt-get -qq update \
        && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
            wireshark-common \
            git \
            tcpdump \
            python3.7-dev \
            gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp
RUN pip install importlib-metadata -r /tmp/requirements.txt

COPY fetch-script /usr/local/bin/


ENTRYPOINT ["fetch-script"]
