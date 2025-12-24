FROM python:3.12

ARG USERNAME=lab
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV USERNAME=$USERNAME

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN python -m pip install --upgrade pip
RUN pip3 install tqdm==4.66.1 trimesh==3.23.5 ruff==0.14.10
RUN pip3 install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

USER $USERNAME