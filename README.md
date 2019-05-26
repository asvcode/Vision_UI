### Visual_UI
Visual UI interface for fastai

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

[![Visual UI Demo](https://j.gifs.com/p8zLzN.gif)](https://www.youtube.com/watch?v=Hdj5NwQrP2o&feature=youtu.be)



Aim to provide an easy to use graphical interface without the need to dig deep into the code.  This visual tool provides a good starting point to get training quickly using fastai.

All tabs are provided within an accordion design using ipywidgets, this allows for all aspects of choosing and viewing parameters in one line of sight

<p align="center">
  <img width="350" height="181" hspace="20" src="static/data.PNG"/><img width="350" height="181" src="static/aug_one.PNG">
</p>

The Augmentation tab utilizes fastai parameters so you can view what different image augmentations look like and compare

<p align="center">
  <img width="350" height="276" hspace="20" src="static/aug_two.PNG"/><img width="316" height="276" src="static/aug_three.PNG">
</p>

View batch information

<p align="center">
  <img width="350" height="305" hspace="20" src="static/batch_two.PNG">
</p>

Review model data and choose suitable metrics for training

<p align="center">
  <img width="332" height="198" hspace="20" src="static/model.PNG"/><img width="350" height="198" src="static/metrics.PNG">
</p>

Review parameters get learning rate and train using the one cycle policy

<p align="center">
  <img width="451" height="184" hspace="20" src="static/LR_one.PNG"/><img width="350" height="184" src="static/LR.PNG">
</p>

Can experiment with various learning rates and train

<p align="center">
  <img width=393" height="273" hspace="20" src="static/LR_three.PNG"/><img width="300" height="273" src="static/Lr_four.PNG">
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


### Installation

git clone this repository

`git clone https://github.com/asvcode/Vision_UI.git`

run `01_UI_Fastai.ipynb` ,specify your path and run `display_ui(path)`


### Known Issues

##### Google Colab

Colab does not currently support ipywidgets because their output is in its own frame so prevents ipywidets from working.  This is the link to issues thread [Link](https://github.com/googlecolab/colabtools/issues/60)

### Future Work

- Currently on works with images using the `ImageDataBunch.from_folder` option.  Plans to expand to `.from_csv` and `.from_df`


[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
