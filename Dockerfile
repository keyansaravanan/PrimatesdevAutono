FROM timongentzsch/l4t-ubuntu20-base

ARG DEBIAN_FRONTEND=noninteractive

ARG BUILD_SOX=1

RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-dev libopenblas-dev libjpeg-dev zlib1g-dev sox libsox-dev libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY assets/*whl /root/


ENV PATH=/usr/local/cuda/bin:/usr/local/cuda-10.2/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-10.2/targets/aarch64-linux/lib:

CMD ["bash"]
WORKDIR /root
