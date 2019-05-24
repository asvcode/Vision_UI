### Visual_UI
Visual UI interface for fastai

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

Aim to provide an easy to use graphical interface

<p align="center">
  <img width="350" height="181" hspace="20" src="static/data.PNG"/><img width="350" height="181" src="static/aug_one.PNG">
</p>

The Augmentation tab utilizes fastai parameters so you can view what different image augmentations look like and compare

<p align="center">
  <img width="350" height="276" hspace="20" src="static/aug_two.PNG"/><img width="316" height="276" src="static/aug_three.PNG">
</p>

Review model data and choose suitable metrics for training

<p align="center">
  <img width="350" height="198" hspace="20" src="static/metrics.PNG"/><img width="332" height="198" src="static/model.PNG">
</p>



### Requirements
- fastai
I am using the developer version:

<p align="left">
  <img width="350" height="100" src="static/info.PNG">
</p>


`git clone https://github.com/fastai/fastai`

`cd fastai`

`tools/run-after-git-clone`

`pip install -e ".[dev]"`

for installation instructions visit [Fastai Installation Readme](https://github.com/fastai/fastai/blob/master/README.md#installation) 

- ipywidgets

`pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension`

or 

`conda install -c conda-forge ipywidgets`

for installation instructions visit [Installation docs](https://ipywidgets.readthedocs.io/en/stable/user_install.html)

- psutil

psutil (process and system utilities) is a cross-platform library for retrieving information on running processes and system utilization (CPU, memory, disks, network, sensors) in Python

`pip install psutil`

#### Known Issues

##### Google Colab

Colab does not currently support ipywidgets because their output is in its own frame so prevents ipywidets from working.  This is the link to issues thread [Link](https://github.com/googlecolab/colabtools/issues/60)
