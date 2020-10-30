#!/bin/bash

wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda"
rm Miniconda3-latest-Linux-x86_64.sh

export PATH="$HOME/miniconda/bin:$PATH"

conda create -y -q --name CSpy3 python=3.6 pip

source activate CSpy3

pip install scipy==1.5.3
pip install Shapely==1.7.1
pip install matplotlib==3.3.2
pip install statsmodels==0.12.1
pip install numdifftools==0.9.39

source deactivate CSpy3

cd $HOME/miniconda/pkgs
rm *.tar.bz2 -f 2> /dev/null
