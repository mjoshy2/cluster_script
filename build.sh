#!/bin/sh
pip install -r requirements.txt
cd src
py credit_data_regularnn.py
py credit_data_clusterednn.py