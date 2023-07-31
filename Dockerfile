# The dockerfile for OpenQAOA braket jobs
FROM 292282985366.dkr.ecr.us-west-2.amazonaws.com/amazon-braket-base-jobs:1.0-cpu-py39-ubuntu22.04

RUN mkdir -p /openqaoa

ADD ./ /openqaoa/

RUN cd openqaoa && make dev-install
