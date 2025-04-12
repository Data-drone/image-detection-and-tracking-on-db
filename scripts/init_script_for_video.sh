#!/bin/bash

# update package repo
sudo apt update

# install os level ffmpeg
sudo apt install -y ffmpeg

# install python lib for all users
/databricks/python/bin/pip install ffmpeg