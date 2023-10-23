FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04
ENV PYTHONUNBUFFERED=1


RUN apt-get update --fix-missing && \
  apt-get install -y wget bzip2 ca-certificates curl git vim-tiny zip unzip \
  file libexpat1-dev libmagic-mgc libmagic1 libmpdec2 libpython3-dev libpython3-stdlib \
  libpython3.8 libpython3.8-dev libpython3.8-minimal libpython3.8-stdlib mime-support \
  python-pip-whl python3 python3-dev python3-distutils python3-lib2to3 python3-minimal \
  python3-pkg-resources python3-setuptools python3-wheel python3.8 python3.8-dev \
  python3.8-minimal python3-pip zlib1g zlib1g-dev rustc cargo && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  bash ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh && \
  /opt/conda/bin/conda clean --all --yes && \
  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate base" >> ~/.bashrc


ENV PATH /opt/conda/bin:$PATH
RUN conda update conda -y
COPY setup.sh /
SHELL ["/bin/bash", "-i", "/setup.sh"]
