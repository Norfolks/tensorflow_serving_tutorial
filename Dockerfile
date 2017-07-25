FROM ubuntu:16.04

MAINTAINER Jeremiah Harmsen <jeremiah@google.com>

RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        mlocate \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev \
        libcurl3-dev \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up grpc

RUN pip install mock grpcio invoke flask tensorflow

# Set up Bazel.

ENV BAZELRC /root/.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.5.1
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download TensorFlow Serving
RUN git clone --recurse-submodules https://github.com/tensorflow/serving && \
  cd serving && \
  git checkout

WORKDIR /serving/tensorflow
RUN tensorflow/tools/ci_build/builds/configured CPU

WORKDIR /serving

RUN bazel build -c opt tensorflow_serving/... 
RUN cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/
RUN mkdir /code && mkdir /code/tensorflow_serving/
RUN cp -r -L bazel-bin/tensorflow_serving/example/inception_client.runfiles/tf_serving/tensorflow_serving/ /code/
RUN bazel clean --expunge

RUN pip install keras Pillow

ADD . /code



WORKDIR /code


CMD ["/bin/bash"]
