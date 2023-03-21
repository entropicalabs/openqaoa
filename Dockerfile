# The dockerfile for OpenQAOA braket jobs
FROM 292282985366.dkr.ecr.us-east-1.amazonaws.com/amazon-braket-pytorch-jobs:1.9.1-gpu-py38-cu111-ubuntu20.04

RUN mkdir -p /openqaoa

ADD ./ /openqaoa/

RUN pip3 install /openqaoa/.
