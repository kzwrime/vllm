#!/bin/bash

pip install -r requirements/build.txt
pip install -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu python setup.py develop