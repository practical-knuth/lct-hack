#!bin/bash

conda create -n lct_env python=3.10.11 -y -q
source activate lct_env
conda activate lct_env
pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -r ./web/requirements.txt