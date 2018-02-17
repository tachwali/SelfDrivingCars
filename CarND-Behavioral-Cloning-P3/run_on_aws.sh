#!/usr/bin/env bash

conda install -c menpo opencv

wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
unzip data.zip

conda env export > aws_environment.yml

python model.py
