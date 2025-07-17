from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets
import ipywidgets as widgets
from IPython.display import display,clear_output

import pandas as pd

from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import*
from fastai.widgets import ClassConfusion

import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askdirectory

import webbrowser
from IPython.display import YouTubeVideo

import warnings
warnings.filterwarnings('ignore')

import xresnet2 #for xrresnet usability

#widget layouts
layout = {'width':'90%', 'height': '50px', 'border': 'solid', 'fontcolor':'lightgreen'}
layout_two = {'width':'100px', 'height': '200px', 'border': 'solid', 'fontcolor':'lightgreen'}
style_green = {'handle_color': 'green', 'readout_color': 'red', 'slider_color': 'blue'}
style_blue = {'handle_color': 'blue', 'readout_color': 'red', 'slider_color': 'blue'}

############################
## Modules for Data 1 tab ##
############################
def dashboard_one():
    """GUI for architecture selection as well as batch size, image size, pre-trained values
    as well as checking system info and links to fastai, fastai forum and asvcode github page"""
    import fastai
    import psutil
    print ('>> Vision_UI Update: 12/23/2019')
    style = {'description_width': 'initial'}

    button = widgets.Button(description='System')
    display(button)

    out = widgets.Output()
    display(out)

    def on_button_clicked_info(b):
        with out:
            clear_output()
            print(f'Fastai Version: {fastai.__version__}')
            print(f'Cuda: {torch.cuda.is_available()}')
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'Python version: {sys.version}')
            print(psutil.cpu_percent())
            print(psutil.virtual_memory())  # physical memory usage
            print('memory % used:', psutil.virtual_memory()[2])

    button.on_click(on_button_clicked_info)

    dashboard_one.norma = widgets.ToggleButtons(
        options=['Imagenet', 'Custom', 'Cifar', 'Mnist'],
        description='Normalization:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Imagenet stats', 'Create your own', 'Cifar stats', 'Mnist stats'],
        style=style
    )
    dashboard_one.archi = widgets.ToggleButtons(
        options=['alexnet', 'BasicBlock', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnet18',
                 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet1_0', 'squeezenet1_1', 'vgg16_bn',
                 'vgg19_bn', 'xresnet18', 'xresnet34', 'xresnet50', 'xresnet101', 'xresnet152'],
        description='Architecture:',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[],
    )
    layout = widgets.Layout(width='auto', height='40px') #set width and height
    dashboard_one.pretrain_check = widgets.Checkbox(
        options=['Yes', "No"],
        description='Pretrained:',
        disabled=False,
        value=True,
        box_style='success',
        button_style='lightgreen', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Default: Checked = use pretrained weights, Unchecked = No pretrained weights'],
    )
    dashboard_one.method = widgets.ToggleButtons(
        options=['cnn_learner', 'unet_learner'],
        description='Method:',
        disabled=True,
        value='cnn_learner',
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Under construction'],
        style=style
    )
    dashboard_one.f=widgets.FloatSlider(min=8,max=64,step=8,value=32, continuous_update=False, layout=layout, style=style_green, description="Batch size")
    dashboard_one.m=widgets.FloatSlider(min=0, max=360, step=16, value=128, continuous_update=False, layout=layout, style=style_green, description='Image size')

    display(dashboard_one.norma, dashboard_one.archi, dashboard_one.pretrain_check, dashboard_one.method, dashboard_one.f, dashboard_one.m)

    print ('>> Resources')
    button_two = widgets.Button(description='Fastai Docs')
    button_three = widgets.Button(description='Fastai Forums')
    button_four = widgets.Button(description='Vision_UI github')

    but_two = widgets.HBox([button_two, button_three, button_four])
    display(but_two)

    def on_doc_info(b):
        webbrowser.open('https://docs.fast.ai/')
    button_two.on_click(on_doc_info)

    def on_forum(b):
        webbrowser.open('https://forums.fast.ai/')
    button_three.on_click(on_forum)

    def vision_utube(b):
            webbrowser.open('https://github.com/asvcode/Vision_UI')
    button_four.on_click(vision_utube)

##################################
## Modules for Augmentation Tab ##
##################################
def dashboard_two():
    """GUI for augmentations"""
    choice_button = widgets.Button(description='Augmentation Image')
    button = widgets.Button(description="View")
    c_button = widgets.Button(description="View Code")

    display(choice_button)

    print('>> Augmentations\n')

    dashboard_two.doflip = widgets.ToggleButtons(
        options=['True', 'False'],
        value=None,
        description='Do Flip:',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )
    dashboard_two.dovert = widgets.ToggleButtons(
        options=['True', "False"],
        value=None,
        description='Do Vert:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )
    dashboard_two.three = widgets.FloatSlider(min=0.1,max=4,step=0.1,value=0.1, description='Max Zoom', orientation='vertical', style=style_green)
    dashboard_two.four = widgets.FloatSlider(min=0.25, max=1.0, step=0.1, value=0.75, description='p_affine', orientation='vertical', style=style_green)
    dashboard_two.five = widgets.FloatSlider(min=0.2,max=0.99, step=0.1,value=0.2, description='Max Lighting', orientation='vertical', style=style_blue)
    dashboard_two.six = widgets.FloatSlider(min=0.25, max=1.1, step=0.1, value=0.75, description='p_lighting', orientation='vertical', style=style_blue)
    dashboard_two.seven = widgets.FloatSlider(min=0.1, max=0.9, step=0.1, value=0.1, description='Max warp', orientation='vertical', style=style_green)

    dashboard_two.cut = widgets.ToggleButtons(
        options=['True', 'False'],
        value='False',
        description='Cutout:',
        disabled=False,
        icon='check',
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )
    dashboard_two.jit = widgets.ToggleButtons(
        options=['True', 'False'],
        value='False',
        description='Jitter:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )
    dashboard_two.contrast = widgets.ToggleButtons(
        options=['True', 'False'],
        description='Contrast:',
        value='False',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    dashboard_two.bright = widgets.ToggleButtons(
        options=['True', 'False'],
        description='Brightness:',
        value='False',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    dashboard_two.rotate = widgets.ToggleButtons(
        options=['True', 'False'],
        value='False',
        description='Rotate:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )
    dashboard_two.sym_warp = widgets.ToggleButtons(
        options=['True', 'False'],
        value='False',
        description='S. Warp:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )
    dashboard_two.pad = widgets.ToggleButtons(
        options=['zeros', 'border', 'reflection'],
        value='reflection',
        description='Padding:',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )

    ui2 = widgets.VBox([dashboard_two.doflip, dashboard_two.dovert])
    ui = widgets.HBox([dashboard_two.three, dashboard_two.seven, dashboard_two.four,dashboard_two.five, dashboard_two.six])
    ui3 = widgets.VBox([ui2, ui])
    ui4 = widgets.VBox([dashboard_two.cut, dashboard_two.jit, dashboard_two.contrast, dashboard_two.bright,
                        dashboard_two.rotate, dashboard_two.sym_warp])
    ui5 = widgets.HBox([ui3, ui4])
    ui6 = widgets.VBox([ui5, dashboard_two.pad])

    display (ui6)

    print ('>> Press button to view augmentations.')
    display(button)

    def on_choice_button(b):
        image_choice()
    choice_button.on_click(on_choice_button)

    out_aug = widgets.Output()
    display(out_aug)

    def on_button_clicked(b):
        with out_aug:
            clear_output()
            additional_aug_choices()

    button.on_click(on_button_clicked)

def cutout_choice():
    """Helper for cutout augmentations"""
    cut_out = dashboard_two.cut.value

    if cut_out == 'True':
        cutout_choice.one = widgets.FloatSlider(min=0,max=10,step=1,value=0, description='Start', orientation='vertical', style=style_green)
        cutout_choice.two = widgets.FloatSlider(min=0, max=40, step=1, value=0, description='End', orientation='vertical', style=style_green)
        cutout_choice.three = widgets.FloatSlider(min=0,max=50, step=1,value=10, description='Length', orientation='vertical', style=style_blue)
        cutout_choice.four = widgets.FloatSlider(min=0, max=50, step=1, value=10, description='Height', orientation='vertical', style=style_blue)

        ui = widgets.HBox([cutout_choice.one, cutout_choice.two, cutout_choice.three, cutout_choice.four])
        print('>> Cutout')
        display(ui)

    else:
        cutout_choice.one = widgets.FloatSlider(value=0)
        cutout_choice.two = widgets.FloatSlider(value=0)
        cutout_choice.three = widgets.FloatSlider(value=0)
        cutout_choice.four = widgets.FloatSlider(value=0)

def jit_choice():
    """Helper for jittery augmentations"""
    jit = dashboard_two.jit.value

    if jit == 'True':
        jit_choice.one = widgets.FloatSlider(min=0.01,max=1.0,step=0.02,value=0.01, description='magnitude', orientation='vertical', style=style_green)
        jit_choice.two = widgets.FloatSlider(min=0, max=1.0, step=0.1, value=0.1, description='p', orientation='vertical', style=style_green)

        ui = widgets.HBox([jit_choice.one, jit_choice.two])
        print('>> Jittery')
        display(ui)

    else:
        jit_choice.one = widgets.FloatSlider(min=0, value=0)
        jit_choice.two = widgets.FloatSlider(min=0, value=0)

def contrast_choice():
    """Helper for contrast augmentations"""
    contr = dashboard_two.contrast.value

    if contr == 'True':
        contrast_choice.one = widgets.FloatSlider(min=0,max=10,step=0.1,value=1, description='scale1', orientation='vertical', style=style_green)
        contrast_choice.two = widgets.FloatSlider(min=0, max=10, step=0.1, value=1, description='scale2', orientation='vertical', style=style_blue)

        ui = widgets.HBox([contrast_choice.one, contrast_choice.two])
        print('>> Contrast')
        display(ui)

    else:
        contrast_choice.one = widgets.FloatSlider(value=1)
        contrast_choice.two = widgets.FloatSlider(value=1)

def bright_choice():
    """Helper for brightness augmentations"""
    bright = dashboard_two.bright.value

    if bright == 'True':
        bright_choice.one = widgets.FloatSlider(min=0,max=1,step=0.1,value=0.1, description='b1', orientation='vertical', style=style_green)
        bright_choice.two = widgets.FloatSlider(min=0, max=1, step=0.1, value=0.1, description='b2', orientation='vertical', style=style_green)
        bright_choice.three = widgets.FloatSlider(min=0, max=1, step=0.1, value=0, description='p', orientation='vertical', style=style_green)

        ui = widgets.HBox([bright_choice.one, bright_choice.two, bright_choice.three])
        print('>> Brightness')
        display(ui)

    else:
        bright_choice.one = widgets.FloatSlider(value=0)
        bright_choice.two = widgets.FloatSlider(value=0)
        bright_choice.three = widgets.FloatSlider(value=0)

def rotate_choice():
    """Helper for rotation augmentations"""
    rotate = dashboard_two.rotate.value

    if rotate == 'True':
        rotate_choice.one = widgets.FloatSlider(min=-90,max=90,step=5,value=-30, description='Degree1', orientation='vertical', style=style_green)
        rotate_choice.two = widgets.FloatSlider(min=-90, max=90, step=5, value=30, description='Degree2', orientation='vertical', style=style_green)
        rotate_choice.three = widgets.FloatSlider(min=0, max=10, step=0.1, value=0, description='p', orientation='vertical', style=style_blue)

        ui = widgets.HBox([rotate_choice.one, rotate_choice.two, rotate_choice.three])
        print('>> Rotate')
        display(ui)

    else:
        rotate_choice.one = widgets.FloatSlider(value=0)
        rotate_choice.two = widgets.FloatSlider(value=0)
        rotate_choice.three = widgets.FloatSlider(value=0)

def sym_w():
    """Helper for Symmetric Warp Augmentations"""
    sym = dashboard_two.sym_warp.value

    if sym == 'True':
        sym_w.one = widgets.FloatSlider(min=-90,max=90,step=5,value=-30, description='mag1', orientation='vertical', style=style_green)
        sym_w.two = widgets.FloatSlider(min=-90, max=90, step=5, value=30, description='mag2', orientation='vertical', style=style_green)
        sym_w.three = widgets.FloatSlider(min=0, max=1, step=0.1, value=0, description='p', orientation='vertical', style=style_blue)

        ui = widgets.HBox([sym_w.one, sym_w.two, sym_w.three])
        print('>> Symmetric Warp')
        display(ui)

    else:
        sym_w.one = widgets.FloatSlider(value=0)
        sym_w.two = widgets.FloatSlider(value=0)
        sym_w.three = widgets.FloatSlider(value=0)

def additional_aug_choices():
    """Helper for additional augmentation choices"""
    cut_c = dashboard_two.cut.value
    jit_c = dashboard_two.jit.value
    cont_c = dashboard_two.contrast.value
    bright_c = dashboard_two.bright.value
    rotate_c = dashboard_two.rotate.value
    sym_c = dashboard_two.sym_warp.value

    if cut_c == 'True':
        cutout_choice()
    else:
        cutout_choice()
    if jit_c == 'True':
        jit_choice()
    else:
        jit_choice()
    if cont_c == 'True':
        contrast_choice()
    else:
        contrast_choice()
    if bright_c == 'True':
        bright_choice()
    else:
        bright_choice()
    if rotate_c == 'True':
        rotate_choice()
    else:
        rotate_choice()
    if sym_c == 'True':
        sym_w()
    else:
        sym_w()

    display_augs(image_choice)

def display_augs(image_path):
    """Helper to display augmentations"""

    button = widgets.Button(description='View Augmentations')
    display(button)
    dis_out = widgets.Output()
    display(dis_out)

    def on_button_clicked(b):
        with dis_out:
            clear_output()

            image_path = image_choice.path
            get_image(image_path)
            image_d = open_image(image_path)
            print(f'Augmentation Image: {image_choice.path}')
            print(f'Image Size: {image_d} \n')
            def get_ex(): return open_image(image_path)

            path = path_choice.path #checkiing to see if needed
            print('Getting augmentations.....')

            #Helper for getting do_flip and flip_vert to work correctly
            flip = ''
            vert = ''

            if dashboard_two.doflip.value == 'True':
                flip = 'True'
            else:
                flip = ''

            if dashboard_two.dovert.value == 'True':
                vert = 'True'
            else:
                vert = ''

            flip_val = dashboard_two.doflip.value #to use for print statements only
            vert_val = dashboard_two.dovert.value #to use for print statements only

            max_zoom=dashboard_two.three.value
            max_warp=dashboard_two.seven.value
            max_light=dashboard_two.five.value
            p_affine=dashboard_two.four.value
            p_light=dashboard_two.six.value
            cut1 = int(cutout_choice.one.value)
            cut2 = int(cutout_choice.two.value)
            length = int(cutout_choice.three.value)
            height = int(cutout_choice.four.value)
            jit1 = float(jit_choice.one.value)
            jit2 = float(jit_choice.two.value)
            scale1 = float(contrast_choice.one.value)
            scale2 = float(contrast_choice.two.value)
            b1 = float(bright_choice.one.value)
            b2 = float(bright_choice.two.value)
            b3 = float(bright_choice.three.value)
            degree1 = float(rotate_choice.one.value)
            degree2 = float(rotate_choice.two.value)
            deg_p = float(rotate_choice.three.value)
            s_war = float(sym_w.one.value)
            s_war2 = float(sym_w.two.value)
            p_sym = float(sym_w.three.value)

            xtra_tfms = [cutout(n_holes=(cut1, cut2), length=(length, height), p=1.), jitter(magnitude=jit1,p=jit2),
                     contrast(scale=(scale1, scale2), p=1.), brightness(change=(b1, b2), p=b3),
                     rotate(degrees=(degree1,degree2), p=deg_p), symmetric_warp(magnitude=(s_war,s_war2), p=p_sym)]

            tfms = get_transforms(do_flip=flip, flip_vert=vert, max_zoom=max_zoom, p_affine=p_affine,
                    max_lighting=max_light, p_lighting=p_light, max_warp=max_warp, xtra_tfms=xtra_tfms)

            print('\n>> Augmentations')
            print(f'do_flip:  {flip_val}| flip_vert: {vert_val}| max_zoom: {max_zoom}| max_warp: {max_warp}| '
            f'p_affine: {p_affine}| max_lighting: {max_light}| p_lighthing: {p_light}')

            print('\n>> Additional Augmentations')
            print(f'cutout(n_holes=({cut1}, {cut2}, length=({length}, {height}), p=1.), jitter(magnitude={jit1},'
            f'p={jit2}), contrast(scale=({scale1}, {scale2}, p=1.), brightness(change=({b1}, {b2}), p={b3}),'
            f'rotate(degrees({degree1}, {degree2}), p={deg_p}), symmetric_warp(magnitude=({s_war}, {s_war2}, p={p_sym})))')

            _, axs = plt.subplots(2,4,figsize=(12,6))
            for ax in axs.flatten():
                img = get_ex().apply_tfms(tfms[0], get_ex(), size=224, padding_mode=dashboard_two.pad.value)
                img.show(ax=ax)
    button.on_click(on_button_clicked)

##########################
##Modules for data 2 tab##
##########################
def get_image(image_path):
    """Get choosen image"""
    #print(image_path)

def path_choice():
    """Choose the data path"""
    root = Tk()
    path_choice.path = askdirectory(title='Select Folder')
    root.destroy()
    path = Path(path_choice.path)
    print('Folder path:', path)
    path_ls = path.ls()
    data_in()
    return path_choice.path

def image_choice():
    """Choose image for augmentations"""
    root = Tk()
    image_choice.path = filedialog.askopenfilename(title='Choose Image')
    root.destroy()
    return image_choice.path

def df_choice():
    """Helper to choose the csv file for using with data in datafolder"""
    root = Tk()
    df_choice.path = filedialog.askopenfilename(title='Choose File')
    root.destroy()
    return df_choice.path

def in_folder_test():
    """Helper to choose folder option """
    root = Tk()
    in_folder_test.path = askdirectory(title='Select Folder')
    root.destroy()
    path = Path(in_folder_test.path)

def in_folder_train():
    """Helper to choose folder option """
    root = Tk()
    in_folder_train.path = askdirectory(title='Select Folder')
    root.destroy()
    path = Path(in_folder_train.path)

def in_folder_valid():
    """Helper to choose folder option """
    root = Tk()
    in_folder_valid.path = askdirectory(title='Select Folder')
    root.destroy()
    path = Path(in_folder_valid.path)

def pct_metrics():
    print('>> Specify train./valid split:')
    pct_metrics.f=widgets.FloatSlider(min=0,max=1,step=0.1,value=0.2, continuous_update=False, style=style_green, description="valid_pct")

    ui2 = widgets.VBox([pct_metrics.f])
    display(ui2)

def csv_folder_choice():
    """Helper to choose folder option """
    root = Tk()
    csv_folder_choice.path = askdirectory(title='Select Folder')
    root.destroy()
    path = Path(csv_folder_choice.path)

def button_f():
    """Helper for folder_choices"""
    print('>> Do you need to specify train and/or valid folder locations:')
    print('>> Leave unchecked for default fastai values')

    button_fs = widgets.Button(description='Confirm')

    button_f.train = widgets.Checkbox(
        value=False,
        description='Specify Train folder location',
        disabled=False
        )
    button_f.valid = widgets.Checkbox(
        value=False,
        description='Specify Valid folder location',
        disabled=False
        )
    ui = widgets.HBox([button_f.train, button_f.valid])
    display(ui)

    display(button_fs)

    out = widgets.Output()
    display(out)

    def on_button_clicked(b):
        with out:
            clear_output()
            folder_choices()
    button_fs.on_click(on_button_clicked)

def folder_choices():
    """Helper for in_folder choices"""
    button_fc = widgets.Button(description='Choice')
    button_tv = widgets.Button(description='Train and Valid folder')
    button_v = widgets.Button(description='Valid Folder')
    button_t = widgets.Button(description='Train folder')

    if button_f.train.value == True and button_f.valid.value == True:
        ui = widgets.HBox([button_tv])

    elif button_f.train.value == False and button_f.valid.value == True:
        ui = widgets.HBox([button_v])

    elif button_f.train.value == True and button_f.valid.value == False:
        ui = widgets.HBox([button_t])

    else:
        ui = None
        print("Using default values of 'train' and 'valid' folders")
        in_folder_train.path = 'train'
        in_folder_valid.path = 'valid'
        pct_metrics()

    out = widgets.Output()
    display(out)

    display(ui)

    def on_button_clicked_tv(b):
        clear_output()
        in_folder_train()
        in_folder_valid()
        print(f'Train folder: {in_folder_train.path}')
        print(f'Valid folder: {in_folder_valid.path}')
        pct_metrics()
    button_tv.on_click(on_button_clicked_tv)

    def on_button_clicked_t(b):
        clear_output()
        in_folder_train()
        print(f'Train folder: {in_folder_train.path}')
        in_folder_valid.path = 'valid'
        pct_metrics()
    button_t.on_click(on_button_clicked_t)

    def on_button_clicked_v(b):
        clear_output()
        in_folder_valid()
        print(f'Valid folder: {in_folder_valid.path}\n')
        in_folder_train.path = 'train'
        pct_metrics()
    button_v.on_click(on_button_clicked_v)

def button_g():
    """Helper for csv_choices"""
    print('Do you need to specify suffix and folder location:')

    button = widgets.Button(description='Confirm')

    button_g.suffix = widgets.Checkbox(
        value=False,
        description='Specify suffix',
        disabled=False
        )
    button_g.folder = widgets.Checkbox(
        value=False,
        description='Specify folder',
        disabled=False
        )
    ui = widgets.HBox([button_g.folder, button_g.suffix])
    display(ui)

    display(button)

    out = widgets.Output()
    display(out)

    def on_button_clicked(b):
        with out:
            clear_output()
            csv_choices()
    button.on_click(on_button_clicked)

def csv_choices():
    """Helper for in_csv choices"""
    print(f'Choose image suffix, location of training folder and csv file')
    button_s = widgets.Button(description='Folder')
    button_f = widgets.Button(description='CSV file')
    button_c = widgets.Button(description='Confirm')

    csv_choices.drop = widgets.Dropdown(
                                        options=[None,'.jpg', '.png', '.jpeg'],
                                        value=None,
                                        description='Suffix:',
                                        disabled=False,
                                        )

    if button_g.folder.value == True and button_g.suffix.value == True:
        ui = widgets.HBox([csv_choices.drop, button_s, button_f])
    elif button_g.folder.value == False and button_g.suffix.value == False:
        ui = widgets.HBox([button_f])
    elif button_g.folder.value == True and button_g.suffix.value == False:
        ui = widgets.HBox([button_s, button_f])
    elif button_g.folder.value == False and button_g.suffix.value == True:
        ui = widgets.HBox([csv_choices.drop, button_f])

    display(ui)
    display(button_c)

    out = widgets.Output()
    display(out)

    def on_button_s(b):
        csv_folder_choice()
    button_s.on_click(on_button_s)

    def on_button_f(b):
        df_choice()
    button_f.on_click(on_button_f)

    def on_button_c(b):

        if button_g.folder.value == True and button_g.suffix.value == True:
            print(csv_folder_choice.path)
            csv_choices.folder_csv = (csv_folder_choice.path.rsplit('/', 1)[1])
            print(f'folder: {csv_choices.folder_csv}\n')
            print(f'CSV file location: {df_choice.path}')
            csv_choices.file_name = (df_choice.path.rsplit('/', 1)[1])
            print (f'CSV file name: {csv_choices.file_name}\n')
            print(f'Image suffix: {csv_choices.drop.value}\n')
        if button_g.folder.value == False and button_g.suffix.value == False:
            print(f'CSV file location: {df_choice.path}')
            csv_choices.folder_csv = 'train'
            csv_choices.file_name = (df_choice.path.rsplit('/', 1)[1])
            print (f'CSV file name: {csv_choices.file_name}\n')
            print(f'Image suffix: {csv_choices.drop.value}\n')
        if button_g.folder.value == True and button_g.suffix.value == False:
            csv_choices.folder_csv = (csv_folder_choice.path.rsplit('/', 1)[1])
            csv_choices.file_name = (df_choice.path.rsplit('/', 1)[1])
        if button_g.folder.value == False and button_g.suffix.value == True:
            csv_choices.folder_csv = 'train'
            csv_choices.file_name = (df_choice.path.rsplit('/', 1)[1])
        pct_metrics()

    button_c.on_click(on_button_c)
def ds():
    """Choose from various data options, either from a custom dataset on computer or from """
    """easy to install datasets"""
    button = widgets.Button(description='Location')

    style = {'description_width': 'initial'}
    ds.datas = widgets.ToggleButtons(
        options=['Custom', 'CATS&DOGS', 'IMAGENETTE',
                 'IMAGENETTE_160', 'IMAGENETTE_320', 'IMAGEWOOF', 'IMAGEWOOF_160', 'IMAGEWOOF_320',
                 'CIFAR', 'CIFAR_100', 'MNIST', 'MNIST_SAMPLE', 'MNIST_TINY', 'FLOWERS', 'FOOD', 'CARS', 'CALTECH' ],
        description='Choose',
        value=None,
        disabled=False,
        button_style='info',
        tooltips=['Choose your folder', ' Cats&Dogs: 25000 images, 819MB',
                  'Imagenette: A subset of 10 easily classified classes from Imagenet, 18000 images, 1.48GB', 'Imagenette_160: 18000 images, 127MB',
                  'Imagenette_320: 18000 images, 358MB', 'ImageWoof: A subset of 10 harder to classify classes from Imagenet, 18000 images, 1.28GB',
                  'ImageWoof_160: 18000 images, 119MB', 'ImageWoof_320: 18000 images, 343MB', 'Cifar: 60000 images, 234MB',
                  'Cifar_100: 100 classes, 60000 images, 234MB', 'Mnist', 'Mnist 14434 images', 'Mnist 1428 images'
                  'A 102 category dataset consisting of 102 flower categories', '101 food categories, with 101,000 images',
                  '16,185 images of 196 classes of cars, Pictures of objects belonging to 101 categories'],
        style=style
    )
    display(ds.datas)

    display(button)

    out_three = widgets.Output()
    display(out_three)

    def on_button_clicked_info2(b):
        with out_three:
            clear_output()
            ds_choice()

    button.on_click(on_button_clicked_info2)

def ds_choice():
    """Helper for dataset choices"""
    print('Choose how the data is saved')
    if ds.datas.value == 'Custom':
        path_choice()
    elif ds.datas.value == 'CATS&DOGS':
        path_choice.path = untar_data(URLs.DOGS)
        data_in()
    elif ds.datas.value == 'IMAGENETTE':
        path_choice.path = untar_data(URLs.IMAGENETTE)
        data_in()
    elif ds.datas.value == 'IMAGENETTE_160':
        path_choice.path = untar_data(URLs.IMAGENETTE_160)
        data_in()
    elif ds.datas.value == 'IMAGENETTE_320':
        path_choice.path = untar_data(URLs.IMAGENETTE_320)
        data_in()
    elif ds.datas.value == 'IMAGEWOOF':
        path_choice.path = untar_data(URLs.IMAGEWOOF)
        data_in()
    elif ds.datas.value == 'IMAGEWOOF_160':
        path_choice.path = untar_data(URLs.IMAGEWOOF_160)
        data_in()
    elif ds.datas.value == 'IMAGEWOOF_320':
        path_choice.path = untar_data(URLs.IMAGEWOOF_320)
        data_in()
    elif ds.datas.value == 'CIFAR':
        path_choice.path = untar_data(URLs.CIFAR)
        data_in()
    elif ds.datas.value == 'CIFAR_100':
        path_choice.path = untar_data(URLs.CIFAR_100)
        data_in()
    elif ds.datas.value == 'MNIST':
        path_choice.path = untar_data(URLs.MNIST)
        data_in()
    elif ds.datas.value == 'MNIST_SAMPLE':
        path_choice.path = untar_data(URLs.MNIST_SAMPLE)
        data_in()
    elif ds.datas.value == 'MNIST_TINY':
        path_choice.path = untar_data(URLs.MNIST_TINY)
        data_in()
    elif ds.datas.value == 'FLOWERS':
        path_choice.path = untar_data(URLs.FLOWERS)
        data_in()
    elif ds.datas.value == 'FOOD':
        path_choice.path = untar_data(URLs.FOOD)
        data_in()
    elif ds.datas.value == 'CARS':
        path_choice.path = untar_data(URLs.CARS)
        data_in()
    elif ds.datas.value == 'CALTECH':
        path_choice.path = untar_data(URLs.CALTECH_101)
        data_in()

def data_in():
    """Helper to determine if the data is in a folder, csv or dataframe"""
    style = {'description_width': 'initial'}

    button = widgets.Button(description='Data In')

    data_in.datain = widgets.ToggleButtons(
        options=['from_folder', 'from_csv'],
        description='Data In:',
        value=None,
        disabled=False,
        button_style='success',
        tooltips=['Data in folder', 'Data in csv format'],
    )
    display(data_in.datain)

    display(button)

    disp_out = widgets.Output()
    display(disp_out)

    def on_choice_button(b):
        with disp_out:
            clear_output()
            if data_in.datain.value == 'from_folder':
                print('From Folder')
                #folder_choices()
                #pct_metrics()
                button_f()
            #TO DO
            #if data_in.datain.value == 'from_df':
            #    print('From DF')
            #    df_choice()
            #    print(f'CSV file location: {df_choice.path}')
            #    file_name = (df_choice.path.rsplit('/', 1)[1])
            #    print (f'CSV file name: {file_name}')
            if data_in.datain.value == 'from_csv':
                print('From CSV')
                button_g()
    button.on_click(on_choice_button)

def get_data():
    """Helper to get the data from the folder, df or csv"""
    if data_in.datain.value == 'from_folder':
        Data_in.in_folder()
    #TO DO
    #elif data_in.datain.value == 'from_df':
    #    Data_in.in_df()
    elif data_in.datain.value == 'from_csv':
        Data_in.in_csv()

class Data_in():
    def in_folder():
        print('\n>> In Folder')
        path = path_choice.path

        """Helpers for correctly selecting transforms and extra transforms"""
        #Helper for getting do_flip and flip_vert to work correctly
        flip = ''
        vert = ''

        if dashboard_two.doflip.value == 'True':
            flip = 'True'
        else:
            flip = ''

        if dashboard_two.dovert.value == 'True':
            vert = 'True'
        else:
            vert = ''

        batch_val = int(dashboard_one.f.value) # batch size
        image_val = int(dashboard_one.m.value) # image size
        flip_val = dashboard_two.doflip.value #to use for print statements only
        vert_val = dashboard_two.dovert.value #to use for print statements only
        max_zoom=dashboard_two.three.value
        max_warp=dashboard_two.seven.value
        max_light=dashboard_two.five.value
        p_affine=dashboard_two.four.value
        p_light=dashboard_two.six.value
        cut1 = int(cutout_choice.one.value)
        cut2 = int(cutout_choice.two.value)
        length = int(cutout_choice.three.value)
        height = int(cutout_choice.four.value)
        jit1 = float(jit_choice.one.value)
        jit2 = float(jit_choice.two.value)
        scale1 = float(contrast_choice.one.value)
        scale2 = float(contrast_choice.two.value)
        b1 = float(bright_choice.one.value)
        b2 = float(bright_choice.two.value)
        b3 = float(bright_choice.three.value)
        degree1 = float(rotate_choice.one.value)
        degree2 = float(rotate_choice.two.value)
        deg_p = float(rotate_choice.three.value)
        s_war = float(sym_w.one.value)
        s_war2 = float(sym_w.two.value)
        p_sym = float(sym_w.three.value)

        r = dashboard_one.pretrain_check.value

        #values for saving model
        value_mone = str(dashboard_one.archi.value)
        value_mtwo = str(dashboard_one.pretrain_check.value)
        value_mthree = str(round(dashboard_one.f.value))
        value_mfour = str(round(dashboard_one.m.value))

        if button_f.train.value == False:
            train_choice = 'train'
        else:
            train_choice = (in_folder_train.path.rsplit('/', 1)[1])
            print(train_choice)

        if button_f.valid.value == False:
            valid_choice = 'valid'
        else:
            valid_choice = (in_folder_valid.path.rsplit('/', 1)[1])

        Data_in.in_folder.from_code = ''

        xtra_tfms = [cutout(n_holes=(cut1, cut2), length=(length, height), p=1.), jitter(magnitude=jit1,p=jit2),
                contrast(scale=(scale1, scale2), p=1.), brightness(change=(b1, b2), p=b3),
                rotate(degrees=(degree1,degree2), p=deg_p), symmetric_warp(magnitude=(s_war,s_war2), p=p_sym)]

        tfms = get_transforms(do_flip=flip, flip_vert=vert, max_zoom=max_zoom, p_affine=p_affine,
                max_lighting=max_light, p_lighting=p_light, max_warp=max_warp, xtra_tfms=xtra_tfms)

        data = ImageDataBunch.from_folder(path,
                                          train=train_choice,
                                          valid=valid_choice,
                                          ds_tfms=tfms,
                                          bs=batch_val,
                                          size=image_val,
                                          valid_pct=pct_metrics.f.value,
                                          padding_mode=dashboard_two.pad.value)

        if display_ui.tab.selected_index == 3: #Batch
            data.show_batch(rows=4, figsize=(10,10))

        #if display_ui.tab.selected_index == 4: #Model


        if display_ui.tab.selected_index == 6 :#Train
            print('FOLDER')
            button_LR = widgets.Button(description='LR')
            button_T = widgets.Button(description='Train')
            disp = widgets.HBox([button_LR, button_T])
            display(disp)

            out_fol = widgets.Output()
            display(out_fol)
            def on_button_clicked(b):
                with out_fol:
                    clear_output()
                    a, b = metrics_list(mets_list, mets_list_code)

                    learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=a, custom_head=None)

                    learn.lr_find()
                    learn.recorder.plot()
            button_LR.on_click(on_button_clicked)

            def on_button_clicked_2(b):
                with out_fol:
                    button = widgets.Button(description='Train_N')
                    clear_output()
                    training_ds()
                    display(button)
                    def on_button_clicked_3(b):
                        lr_work()
                        a, b = metrics_list(mets_list, mets_list_code)
                        b_ = b[0]
                        learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=a, custom_head=None)
                        print(f'Training in folder......{b}')
                        cycle_l = int(training_ds.cl.value)

                        #save model
                        file_model_name = value_mone + '_pretrained_' + value_mtwo + '_batch_' + value_mthree + '_image_' + value_mfour

                        learn.fit_one_cycle(cycle_l,
                                            slice(lr_work.info),
                                            callbacks=[SaveModelCallback(learn, every='improvement', monitor=b_, name='best_'+ file_model_name)])
                        learn.save(file_model_name)

                    button.on_click(on_button_clicked_3)
            button_T.on_click(on_button_clicked_2)

        if display_ui.tab.selected_index == 8: #Code
            print(f'data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs={batch_val}, size={image_val},'
                        f'padding_mode={dashboard_two.pad.value})')

    def in_df():
        print('\n>> In Data frame')
        path = path_choice.path

        """Helpers for correctly selecting transforms and extra transforms"""
        #Helper for getting do_flip and flip_vert to work correctly
        flip = ''
        vert = ''

        if dashboard_two.doflip.value == 'True':
            flip = 'True'
        else:
            flip = ''

        if dashboard_two.dovert.value == 'True':
            vert = 'True'
        else:
            vert = ''
        batch_val = int(dashboard_one.f.value) # batch size
        image_val = int(dashboard_one.m.value) # image size
        flip_val = dashboard_two.doflip.value #to use for print statements only
        vert_val = dashboard_two.dovert.value #to use for print statements only
        max_zoom=dashboard_two.three.value
        max_warp=dashboard_two.seven.value
        max_light=dashboard_two.five.value
        p_affine=dashboard_two.four.value
        p_light=dashboard_two.six.value
        cut1 = int(cutout_choice.one.value)
        cut2 = int(cutout_choice.two.value)
        length = int(cutout_choice.three.value)
        height = int(cutout_choice.four.value)
        jit1 = float(jit_choice.one.value)
        jit2 = float(jit_choice.two.value)
        scale1 = float(contrast_choice.one.value)
        scale2 = float(contrast_choice.two.value)
        b1 = float(bright_choice.one.value)
        b2 = float(bright_choice.two.value)
        b3 = float(bright_choice.three.value)
        degree1 = float(rotate_choice.one.value)
        degree2 = float(rotate_choice.two.value)
        deg_p = float(rotate_choice.three.value)
        s_war = float(sym_w.one.value)
        s_war2 = float(sym_w.two.value)
        p_sym = float(sym_w.three.value)

        xtra_tfms = [cutout(n_holes=(cut1, cut2), length=(length, height), p=1.), jitter(magnitude=jit1,p=jit2),
                contrast(scale=(scale1, scale2), p=1.), brightness(change=(b1, b2), p=b3),
                rotate(degrees=(degree1,degree2), p=deg_p), symmetric_warp(magnitude=(s_war,s_war2), p=p_sym)]

        tfms = get_transforms(do_flip=flip, flip_vert=vert, max_zoom=max_zoom, p_affine=p_affine,
                max_lighting=max_light, p_lighting=p_light, max_warp=max_warp, xtra_tfms=xtra_tfms)

        df = pd.read_csv(df_choice.path)

        data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, bs=batch_val, size=image_val,
                                      padding_mode=dashboard_two.pad.value)
        data.normalize(stats_info())

        if display_ui.tab.selected_index == 3: #Batch
            data.show_batch(rows=4, figsize=(10,10))

        if display_ui.tab.selected_index == 8: #Code

            print(f'df = pd.read_csv({df_choice.path})')
            print(f'\nImageDataBunch.from_df(path, df, ds_tfms=tfms, bs={batch_val}, size={image_val},'
                        f'padding_mode={dashboard_two.pad.value}), valid_pct={pct_metrics.f.value}')

    def in_csv():
        print('\n>> In CSV')
        path = path_choice.path

        """Helpers for correctly selecting transforms and extra transforms"""
        #Helper for getting do_flip and flip_vert to work correctly
        flip = ''
        vert = ''

        if dashboard_two.doflip.value == 'True':
            flip = 'True'
        else:
            flip = ''

        if dashboard_two.dovert.value == 'True':
            vert = 'True'
        else:
            vert = ''
        batch_val = int(dashboard_one.f.value) # batch size
        image_val = int(dashboard_one.m.value) # image size
        flip_val = dashboard_two.doflip.value #to use for print statements only
        vert_val = dashboard_two.dovert.value #to use for print statements only
        max_zoom=dashboard_two.three.value
        max_warp=dashboard_two.seven.value
        max_light=dashboard_two.five.value
        p_affine=dashboard_two.four.value
        p_light=dashboard_two.six.value
        cut1 = int(cutout_choice.one.value)
        cut2 = int(cutout_choice.two.value)
        length = int(cutout_choice.three.value)
        height = int(cutout_choice.four.value)
        jit1 = float(jit_choice.one.value)
        jit2 = float(jit_choice.two.value)
        scale1 = float(contrast_choice.one.value)
        scale2 = float(contrast_choice.two.value)
        b1 = float(bright_choice.one.value)
        b2 = float(bright_choice.two.value)
        b3 = float(bright_choice.three.value)
        degree1 = float(rotate_choice.one.value)
        degree2 = float(rotate_choice.two.value)
        deg_p = float(rotate_choice.three.value)
        s_war = float(sym_w.one.value)
        s_war2 = float(sym_w.two.value)
        p_sym = float(sym_w.three.value)

        r = dashboard_one.pretrain_check.value

        #values for saving model
        value_mone = str(dashboard_one.archi.value)
        value_mtwo = str(dashboard_one.pretrain_check.value)
        value_mthree = str(round(dashboard_one.f.value))
        value_mfour = str(round(dashboard_one.m.value))

        xtra_tfms = [cutout(n_holes=(cut1, cut2), length=(length, height), p=1.), jitter(magnitude=jit1,p=jit2),
                contrast(scale=(scale1, scale2), p=1.), brightness(change=(b1, b2), p=b3),
                rotate(degrees=(degree1,degree2), p=deg_p), symmetric_warp(magnitude=(s_war,s_war2), p=p_sym)]

        tfms = get_transforms(do_flip=flip, flip_vert=vert, max_zoom=max_zoom, p_affine=p_affine,
                max_lighting=max_light, p_lighting=p_light, max_warp=max_warp, xtra_tfms=xtra_tfms)

        label_csv = (df_choice.path)

        if button_g.folder.value == True and button_g.suffix.value == True:
            csv_choices.folder_csv = str(csv_folder_choice.path.rsplit('/', 1)[1])
        elif button_g.folder.value == False and button_g.suffix.value == False:
            csv_choices.folder_csv = None

        data = ImageDataBunch.from_csv(path,
                                       folder=csv_choices.folder_csv,
                                       bs=batch_val,
                                       size=image_val,
                                       suffix=csv_choices.drop.value,
                                       csv_labels=csv_choices.file_name,
                                       label_delim = ' ',
                                       ds_tfms=tfms,
                                       valid_pct=pct_metrics.f.value,
                                       padding_mode=dashboard_two.pad.value)
        data.normalize(stats_info())

        if display_ui.tab.selected_index == 3 :#Batch
            out_batch = widgets.Output()
            display(out_batch)
            with out_batch:
                clear_output()
                data.show_batch(rows=4, figsize=(10,10))

        if display_ui.tab.selected_index == 6 :#Train
            button_LR = widgets.Button(description='LR')
            button_T = widgets.Button(description='Train')
            disp = widgets.HBox([button_LR, button_T])
            display(disp)

            out_csv = widgets.Output()
            display(out_csv)
            def on_button_clicked(b):
                with out_csv:
                    clear_output()
                    a, b = metrics_list(mets_list, mets_list_code)
                    learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=a, custom_head=None)

                    learn.lr_find()
                    learn.recorder.plot()
            button_LR.on_click(on_button_clicked)

            def on_button_clicked_2(b):
                with out_csv:
                    button = widgets.Button(description='Train_N')
                    clear_output()
                    training_ds()
                    display(button)
                    def on_button_clicked_3(b):
                        lr_work()
                        a, b = metrics_list(mets_list, mets_list_code)
                        b_ = b[0]
                        learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=a, custom_head=None)
                        print(f'Training in csv......{b}')
                        print(a)
                        print(b)
                        print(b_)
                        cycle_l = int(training_ds.cl.value)

                        #save model
                        file_model_name = value_mone + '_pretrained_' + value_mtwo + '_batch_' + value_mthree + '_image_' + value_mfour

                        learn.fit_one_cycle(cycle_l, slice(lr_work.info),
                        callbacks=[SaveModelCallback(learn, every='improvement', monitor=b_, name='best_' + file_model_name)])

                        learn.save(file_model_name)

                    button.on_click(on_button_clicked_3)
            button_T.on_click(on_button_clicked_2)

        if display_ui.tab.selected_index == 8: #Code

            file_name = (df_choice.path.rsplit('/', 1)[1])
            print(f"data = ImageDataBunch.from_csv(path, folder='{csv_choices.folder_csv}', bs={batch_val}, size={image_val}, suffix={csv_choices.drop.value},"
                 f"csv_labels='{file_name}', label_delim='', ds_tfms=tfms, padding_mode={dashboard_two.pad.value}), valid_pct={pct_metrics.f.value}")

##########################
## Modules for Model tab##
##########################
def model_summary():
    print('>> Review Model information: ', dashboard_one.archi.value)

    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size

    button_summary = widgets.Button(description="Model Summary")
    button_model_0 = widgets.Button(description='Model[0]')
    button_model_1 = widgets.Button(description='Model[1]')

    tfms = get_transforms(do_flip=dashboard_two.doflip.value, flip_vert=dashboard_two.dovert.value, max_zoom=dashboard_two.three.value,
                          p_affine=dashboard_two.four.value, max_lighting=dashboard_two.five.value, p_lighting=dashboard_two.six.value,
                          max_warp=dashboard_two.seven.value, xtra_tfms=None)

    path = path_choice.path
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val)

    r = dashboard_one.pretrain_check.value

    ui_out = widgets.HBox([button_summary, button_model_0, button_model_1])

    arch_work()

    display(ui_out)
    out = widgets.Output()
    display(out)

    def on_button_clicked_summary(b):
        with out:
            clear_output()
            print('working...''\n')
            learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, custom_head=None)
            print('Model Summary')
            info = learn.summary()
            print(info)

    button_summary.on_click(on_button_clicked_summary)

    def on_button_clicked_model_0(b):
        with out:
            clear_output()
            print('working...''\n')
            learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, custom_head=None)
            print('Model[0]')
            info_s = learn.model[0]
            print(info_s)

    button_model_0.on_click(on_button_clicked_model_0)

    def on_button_clicked_model_1(b):
        with out:
            clear_output()
            print('working...''\n')
            learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, custom_head=None)
            print('Model[1]')
            info_sm = learn.model[1]
            print(info_sm)

    button_model_1.on_click(on_button_clicked_model_1)

def arch_work():
    arch_map = {
        'alexnet': models.alexnet,
        'BasicBlock': models.BasicBlock,
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'squeezenet1_0': models.squeezenet1_0,
        'squeezenet1_1': models.squeezenet1_1,
        'vgg16_bn': models.vgg16_bn,
        'vgg19_bn': models.vgg19_bn,
        # 'wrn_22': models.wrn_22,
        'xresnet18': xresnet2.xresnet18,  # using xresnet2.py
        'xresnet34': xresnet2.xresnet34_2,  # using xresnet2.py
        'xresnet50': xresnet2.xresnet50_2,  # using xresent2.py
        'xresnet101': xresnet2.xresnet101,  # using xresent2.py
        'xresnet152': xresnet2.xresnet152,  # using xresnet2.py
    }

    arch_work.code_i = str(dashboard_one.archi.value)
    arch_work.info = arch_map.get(arch_work.code_i)

    output = arch_work.info
    output_c = arch_work.code_i

##########################
## Code for metrics tab ##
##########################
mets_list = []
mets_list_code = []

def metrics_dashboard():
    """Metrics dashboard"""
    button = widgets.Button(description="Metrics")

    metrics_dashboard.error_choice = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Error Choice:',
        value='No',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    metrics_dashboard.accuracy = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Accuracy:',
        value='No',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    metrics_dashboard.accuracy_thresh = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Accuracy Threshs:',
        value='No',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    metrics_dashboard.topk = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Top K:',
        value='No',
        disabled=False,
        button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    metrics_dashboard.recall = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Recall:',
        value='No',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    metrics_dashboard.precision = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Precision:',
        value='No',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    metrics_dashboard.dice = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Dice:',
        value='No',
        disabled=False,
        button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[''],
    )
    layout = widgets.Layout(width='auto', height='40px') #set width and height

    centre_t = widgets.Button(
        description='',
        disabled=True,
        display='flex',
        flex_flow='column',
        align_items='stretch',
        layout = layout
)
    ui = widgets.HBox([metrics_dashboard.error_choice, metrics_dashboard.accuracy, metrics_dashboard.accuracy_thresh, metrics_dashboard.topk])
    ui2 = widgets.HBox([metrics_dashboard.recall, metrics_dashboard.precision, metrics_dashboard.dice])
    ui3 = widgets.VBox([ui, centre_t, ui2])

    r = dashboard_one.pretrain_check.value
    display(ui3)

    print('>> Click to view choosen metrics')
    display(button)

    out = widgets.Output()
    display(out)

    def on_button_clicked(b):
        with out:
            clear_output()
            a, b = metrics_list(mets_list, mets_list_code)
            print('Training Metrics''\n')
            print(a, b)
    button.on_click(on_button_clicked)

def metrics_list(mets_list, mets_list_code):
    """Helper for metrics tab"""
    mets_error = metrics_dashboard.error_choice.value
    mets_accuracy= metrics_dashboard.accuracy.value
    mets_accuracy_thr = metrics_dashboard.topk.value
    mets_accuracy_thresh = metrics_dashboard.accuracy_thresh.value
    mets_precision = metrics_dashboard.precision.value
    mets_recall = metrics_dashboard.recall.value
    mets_dice = metrics_dashboard.dice.value

    acc_code = str('accuracy')
    err_code = str('error_rate')
    thr_code = str('accuracy_thresh')
    k_code = str('top_k_accuracy')
    pre_code = str('precision')
    rec_code = str('recall')
    dice_code = str('dice')

    mets_list=[]
    mets_list_code = []
    output_pres = Precision()
    output_recall = Recall()

    if mets_error == 'Yes':
        mets_list.append(error_rate)
        mets_list_code.append(err_code)
    else:
        None
    if mets_accuracy == 'Yes':
        mets_list.append(accuracy)
        mets_list_code.append(acc_code)
    else:
        None
    if mets_accuracy_thresh == 'Yes':
        mets_list.append(accuracy_thresh)
        mets_list_code.append(thr_code)
    else:
        None
    if mets_accuracy_thr == 'Yes':
        k = data.c
        mets_list.append(top_k_accuracy)
        mets_list_code.append(k_code)
    else:
        None
    if mets_precision == 'Yes':
        mets_list.append(output_pres)
        mets_list_code.append(pre_code)
    else:
        None
    if mets_recall == 'Yes':
        mets_list.append(output_recall)
        mets_list_code.append(rec_code)
    else:
        None
    if mets_dice == 'Yes':
        mets_list.append(dice)
        mets_list_code.append(dice_code)
    else:
        None

    return mets_list, mets_list_code

###########################
## Modules for train tab ##
###########################
def stats_info():

    if dashboard_one.norma.value == 'Imagenet':
        stats_info.stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif dashboard_one.norma.value == 'Cifar':
        stats_info.stats = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
    elif dashboard_one.norma.value == 'Mnist':
        stats_info.stats = ([0.15, 0.15, 0.15], [0.15, 0.15, 0.15])
    else: # dashboard_one.norma.value == 'Custom':
        stats_info.stats = None

    stats = stats_info.stats

def info_lr():
    flip = ''
    vert = ''

    if dashboard_two.doflip.value == 'True':
        flip = 'True'
    else:
        flip = 'False'

    if dashboard_two.dovert.value == 'True':
        vert = 'True'
    else:
        vert = 'False'
    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size
    flip_val = dashboard_two.doflip.value #to use for print statements only
    vert_val = dashboard_two.dovert.value #to use for print statements only
    max_zoom=dashboard_two.three.value
    max_warp=dashboard_two.seven.value
    max_light=dashboard_two.five.value
    p_affine=dashboard_two.four.value
    p_light=dashboard_two.six.value
    cut1 = int(cutout_choice.one.value)
    cut2 = int(cutout_choice.two.value)
    length = int(cutout_choice.three.value)
    height = int(cutout_choice.four.value)
    jit1 = float(jit_choice.one.value)
    jit2 = float(jit_choice.two.value)
    scale1 = float(contrast_choice.one.value)
    scale2 = float(contrast_choice.two.value)
    b1 = float(bright_choice.one.value)
    b2 = float(bright_choice.two.value)
    b3 = float(bright_choice.three.value)
    degree1 = float(rotate_choice.one.value)
    degree2 = float(rotate_choice.two.value)
    deg_p = float(rotate_choice.three.value)
    s_war = float(sym_w.one.value)
    s_war2 = float(sym_w.two.value)
    p_sym = float(sym_w.three.value)
    button = widgets.Button(description='Review Parameters')
    button_two = widgets.Button(description='LR')
    button_three = widgets.Button(description='Train')

    butlr = widgets.HBox([button, button_two, button_three])
    display(butlr)

    out = widgets.Output()
    display(out)

    def on_button_clicked_info(b):
        with out:
            clear_output()

            print(f'Data in: {ds.datas.value}| Normalization: {dashboard_one.norma.value}| Architecture: {dashboard_one.archi.value}| Pretrain: {dashboard_one.pretrain_check.value}|'
            f'Batch Size: {dashboard_one.f.value}| Image Size: {dashboard_one.m.value}')

            print('\n>> Augmentations')
            print(f'do_flip:  {flip}| flip_vert: {vert}| max_zoom: {max_zoom}| max_warp: {max_warp}| '
            f'p_affine: {p_affine}| max_lighting: {max_light}| p_lighthing: {p_light}')

            print('\n>> Additional Augmentations')
            print(f'cutout(n_holes=({cut1}, {cut2}, length=({length}, {height}), p=1.), jitter(magnitude={jit1},'
            f'p={jit2}), contrast(scale=({scale1}, {scale2}, p=1.), brightness(change=({b1}, {b2}), p={b3}),'
            f'rotate(degrees({degree1}, {degree2}), p={deg_p}))')
            print(f'\n Padding: {dashboard_two.pad.value}')

            print(f'Training Metrics: {metrics_list(mets_list)} ')

    button.on_click(on_button_clicked_info)

    def on_button_clicked_info2(b):
        with out:
            clear_output()
            learn_dash()

    button_two.on_click(on_button_clicked_info2)

    def on_button_clicked_info3(b):
        with out:
            clear_output()
            print('Train')
            training()

    button_three.on_click(on_button_clicked_info3)

def lr_work():
    if training_ds.lr.value == '1e-6':
        lr_work.info = float(0.000001)
    elif training_ds.lr.value == '1e-5':
        lr_work.info = float(0.00001)
    elif training_ds.lr.value == '1e-4':
        lr_work.info = float(0.0001)
    elif training_ds.lr.value == '1e-3':
        lr_work.info = float(0.001)
    elif training_ds.lr.value == '1e-2':
        lr_work.info = float(0.01)
    elif training_ds.lr.value == '1e-1':
        lr_work.info = float(0.1)

def training_ds():
    print(">> Using fit_one_cycle \n >> Model saved as ('architecture' + 'pretrained' + batchsize + image size) in model path")
    print(">> Best model also saved as (best_'architecture' + 'pretrained' + batchsize + image size)")
    button = widgets.Button(description='Train')

    style = {'description_width': 'initial'}

    training_ds.cl=widgets.FloatSlider(min=1,max=64,step=1,value=1, continuous_update=False, layout=layout, style=style_green, description="Cycle Length")
    training_ds.lr = widgets.ToggleButtons(
        options=['1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1'],
        description='Learning Rate:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        style=style,
        value='1e-2',
        tooltips=['Choose a suitable learning rate'],
    )

    display(training_ds.cl, training_ds.lr)

def learn_dash():
    button = widgets.Button(description="Learn")
    print ('Choosen metrics: ',metrics_list(mets_list))
    metrics_list(mets_list)

    flip = ''
    vert = ''

    if dashboard_two.doflip.value == 'True':
        flip = 'True'
    else:
        flip = 'False'

    if dashboard_two.dovert.value == 'True':
        vert = 'True'
    else:
        vert = 'False'

    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size

    r = dashboard_one.pretrain_check.value
    t = metrics_list(mets_list)

    flip_val = dashboard_two.doflip.value #to use for print statements only
    vert_val = dashboard_two.dovert.value #to use for print statements only
    max_zoom=dashboard_two.three.value
    max_warp=dashboard_two.seven.value
    max_light=dashboard_two.five.value
    p_affine=dashboard_two.four.value
    p_light=dashboard_two.six.value
    cut1 = int(cutout_choice.one.value)
    cut2 = int(cutout_choice.two.value)
    length = int(cutout_choice.three.value)
    height = int(cutout_choice.four.value)
    jit1 = float(jit_choice.one.value)
    jit2 = float(jit_choice.two.value)
    scale1 = float(contrast_choice.one.value)
    scale2 = float(contrast_choice.two.value)
    b1 = float(bright_choice.one.value)
    b2 = float(bright_choice.two.value)
    b3 = float(bright_choice.three.value)
    degree1 = float(rotate_choice.one.value)
    degree2 = float(rotate_choice.two.value)
    deg_p = float(rotate_choice.three.value)
    s_war = float(sym_w.one.value)
    s_war2 = float(sym_w.two.value)
    p_sym = float(sym_w.three.value)

    xtra_tfms = [cutout(n_holes=(cut1, cut2), length=(length, height), p=1.), jitter(magnitude=jit1,p=jit2),
                contrast(scale=(scale1, scale2), p=1.), brightness(change=(b1, b2), p=b3),
                rotate(degrees=(degree1,degree2), p=deg_p), symmetric_warp(magnitude=(s_war,s_war2), p=p_sym)]

    tfms = get_transforms(do_flip=flip, flip_vert=vert, max_zoom=max_zoom, p_affine=p_affine,
                max_lighting=max_light, p_lighting=p_light, max_warp=max_warp, xtra_tfms=xtra_tfms)

    path = path_choice.path
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')

    learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=metrics_list(mets_list,mets_list_code), custom_head=None)

    learn.lr_find()
    learn.recorder.plot()

#############################
## Modules for Results tab ##
#############################
def path_choice_two():
    root = Tk()
    path_choice_two.path = askdirectory(title='Select Folder')
    root.destroy()
    print('Data folder:', {path_choice_two.path})
    model_choice()

def model_choice():
    root = Tk()
    model_choice.path = filedialog.askopenfilename(title='Select .pth file to load')
    print('File to be loaded: ', {model_choice.path})
    root.destroy()

def load_model():
    load_model.model_path_2 = model_choice.path.split('.pth')
    load_model.model_path_2a = load_model.model_path_2[0]
    loading_model()

def arch_choice():
    print('\n''>> Choose Model architecture and pretrained value of trained model (to match saved model)' '\n' '>> Then load model''\n')
    style = {'description_width': 'initial'}

    arch_choice.archi = widgets.ToggleButtons(
        options=['alexnet', 'BasicBlock', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnet18',
                 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet1_0', 'squeezenet1_1', 'vgg16_bn',
                 'vgg19_bn', 'xresnet18', 'xresnet34', 'xresnet50', 'xresnet101', 'xresnet152'],
        description='Architecture:',
        disabled=False,
        button_style='info',
        tooltips=[],
    )
    layout = widgets.Layout(width='auto', height='40px') #set width and height

    arch_choice.pretrain_check = widgets.Checkbox(
        options=['Yes', "No"],
        description='Pretrained:',
        disabled=False,
        value=True,
        box_style='success',
        button_style='lightgreen',
        tooltips=['Default: Checked = use pretrained weights, Unchecked = No pretrained weights'],
    )

    display(arch_choice.archi, arch_choice.pretrain_check)
    load_model()

def loading_model():
    loading_button = widgets.Button(description='Load Model')

    print('You need a <test> folder for calculating interpretations ')

    display(loading_button)
    def on_loading_clicked(b):
        arch_working()
        print('>> Arch', arch_working.info)
        print('>> Model Name', load_model.model_path_2a)

        tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.25, max_zoom=1.07,
                   max_lighting=0.2, max_warp=0.1, p_affine=0.2,
                   p_lighting=0.2, xtra_tfms=None)

        data = ImageDataBunch.from_folder(path_choice_two.path, valid_pct=0.2,
                                       ds_tfms=tfms, bs=32,
                                       size=128)

        loading_model.learn = cnn_learner(data, base_arch=arch_working.info, pretrained=arch_choice.pretrain_check.value, custom_head=None)
        loading_model.learn.load(load_model.model_path_2a)
        print('>> Model loaded')
        print('>> Getting Intepretations....')
        inference()
        print('Done')
        rs()

    loading_button.on_click(on_loading_clicked)

def arch_working():
    if arch_choice.archi.value == 'alexnet':
        arch_working.info = models.alexnet
    elif arch_choice.archi.value == 'BasicBlock':
        arch_working.info = models.BasicBlock
    elif arch_choice.archi.value == 'densenet121':
        arch_working.info = models.densenet121
    elif arch_choice.archi.value == 'densenet161':
        arch_working.info = models.densenet161
    elif arch_choice.archi.value == 'densenet169':
        arch_working.info = models.densenet169
    elif arch_choice.archi.value == 'densenet201':
        arch_working.info = models.densenet201
    if arch_choice.archi.value == 'resnet18':
        arch_working.info = models.resnet18
    elif arch_choice.archi.value == 'resnet34':
        arch_working.info = models.resnet34
    elif arch_choice.archi.value == 'resnet50':
        arch_working.info = models.resnet50
    elif arch_choice.archi.value == 'resnet101':
        arch_working.info = models.resnet101
    elif arch_choice.archi.value == 'resnet152':
        arch_working.info = models.resnet152
    elif arch_choice.archi.value == 'squeezenet1_0':
        arch_working.info = models.squeezenet1_0
    elif arch_choice.archi.value == 'squeezenet1_1':
        arch_working.info = models.squeezenet1_1
    elif arch_choice.archi.value == 'vgg16_bn':
        arch_working.info = models.vgg16_bn
    elif arch_choice.archi.value == 'vgg19_bn':
        arch_working.info = models.vgg19_bn
    #elif dashboard_one.archi.value == 'wrn_22':
    #    arch_work.info = models.wrn_22
    elif arch_choice.archi.value == 'xresnet18':
        arch_working.info = xresnet2.xresnet18 #using xresnet2.py
    elif arch_choice.archi.value == 'xresnet34':
        arch_working.info = xresnet2.xresnet34_2 #using xresnet2.py
    elif arch_choice.archi.value == 'xresnet50':
        arch_working.info = xresnet2.xresnet50_2 #using xresent2.py
    elif arch_choice.archi.value == 'xresnet101':
        arch_working.info = xresnet2.xresnet101 #using xresent2.py
    elif arch_choice.archi.value == 'xresnet152':
        arch_working.info = xresnet2.xresnet152 #using xresnet2.py

    output = arch_working.info
    output
    print(output)

def inference():
    preds,y,losses = loading_model.learn.get_preds(with_loss=True)
    inference.interp = ClassificationInterpretation(loading_model.learn, preds, y, losses)
    losses, ids = inference.interp.top_losses(inference.interp.data.c)

    _, inference.tl_idx = inference.interp.top_losses(len(inference.interp.losses))

def rs():
    print('>> Model loaded: ', model_choice.path)
    print('>> Use options below to view results')

    plot_button = widgets.Button(description='Multi_Top_Losses')
    cmap_button = widgets.Button(description='Top_Losses')
    cm_button = widgets.Button(description='Confusion Matrix')
    cf_button = widgets.Button(description='Class Confusion')

    dip = widgets.HBox([cm_button, plot_button, cmap_button, cf_button])

    display(dip)

    out = widgets.Output()
    display(out)

    def on_plot_button(b):
        with out:
            clear_output()
            inference.interp.plot_multi_top_losses(9, figsize=(7,7))
    plot_button.on_click(on_plot_button)

    def on_cmap_button(b):
        with out:
            clear_output()
            inference.interp.plot_top_losses(12)
    cmap_button.on_click(on_cmap_button)

    def on_cm_button(b):
        with out:
            clear_output()
            confusion_test()
    cm_button.on_click(on_cm_button)

    def on_cf_button(b):
        with out:
            clear_output()
            print('working...')
            class_f()
    cf_button.on_click(on_cf_button)

def class_f():
    """Helper for class confusion"""
    button = widgets.Button(description='View')
    tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.25, max_zoom=1.07,
                   max_lighting=0.2, max_warp=0.1, p_affine=0.2,
                   p_lighting=0.2, xtra_tfms=None)

    data = ImageDataBunch.from_folder(path_choice_two.path, valid='test',
                                       ds_tfms=tfms, bs=128,
                                       size=128)

    li = data.classes

    class_f.cf_class = widgets.SelectMultiple(
        options=li,
        description='Classes',
        rows = len(li),
        disabled=False
    )
    class_f.cl=widgets.FloatSlider(min=1,max=20,step=1,value=1, disabled=True, continuous_update=False, style=style_green, description="k value")

    ui_f = widgets.HBox([class_f.cf_class, class_f.cl])
    display(ui_f)

    out= widgets.Output()
    display(button)
    display(out)
    def on_button_f(b):
        with out:
            clear_output()
            print(class_f.cf_class.value)
            print(class_f.cl.value)
            classlist = class_f.cf_class.value
            print(classlist)
            ClassConfusion(inference.interp, classlist, is_ordered=False, figsize=(8,8))
    button.on_click(on_button_f)

def confusion_test():
    print('working.......')
    tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.25, max_zoom=1.07,
                   max_lighting=0.2, max_warp=0.1, p_affine=0.2,
                   p_lighting=0.2, xtra_tfms=None)

    data = ImageDataBunch.from_folder(path_choice_two.path, valid='test',
                                       ds_tfms=tfms, bs=128,
                                       size=128)

    class_num = data.c
    print(f'Number of classes: {class_num}')

    if class_num == 2:
        heatmap_choice()
    elif class_num > 2:
        print('More than 2 classes - unable to split into TN, FN, TP, FP')
        inference.interp.plot_confusion_matrix(figsize=(5,5))

def confusion():
    print ('>> Confusion Matrix')

    upp, low = inference.interp.confusion_matrix()
    tn, fp = upp[0], upp[1]
    fn, tp = low[0], low[1]

    button = widgets.Button(description=f'True Negative ({tn})')
    button_two = widgets.Button(description=f'True Positive ({tp})')
    button_three = widgets.Button(description=f'False Positive ({fp})')
    button_four = widgets.Button(description=f'False Negative ({fn})')

    cm_one = widgets.HBox([button, button_three])
    cm_two = widgets.HBox([button_four, button_two])
    cm_three = widgets.VBox([cm_one, cm_two])
    display(cm_three)

    out = widgets.Output()
    display(out)

    def on_tn_button(b):
        with out:
            clear_output()
            print('True Negative')
            TN()
    button.on_click(on_tn_button)

    def on_fp_button(b):
        with out:
            clear_output()
            print('False Positive')
            FP()
    button_three.on_click(on_fp_button)

    def on_fn_button(b):
        with out:
            clear_output()
            print('False Negative')
            FN()
    button_four.on_click(on_fn_button)

    def on_tp_button(b):
        with out:
            clear_output()
            print('True Positive')
            TP()
    button_two.on_click(on_tp_button)

    inference.interp.plot_confusion_matrix(figsize=(5,5))

def heatmap_choice():
    print ('>>  View Heatmaps? \n')

    hmc_button = widgets.Button(description='Click Here')

    heatmap_choice.values = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='',
        disabled=False,
        value=None,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[],
    )

    heat_one = widgets.HBox([heatmap_choice.values, hmc_button])
    display(heat_one)

    out = widgets.Output()
    display(out)

    def on_hmc_button(b):
        clear_output()
        if heatmap_choice.values.value == 'Yes':
            heatmap_choice.choice = False
            cm_values()
        else:
            heatmap_choice.choice = True
            confusion()
    hmc_button.on_click(on_hmc_button)

def cm_values():
    cm_values.color = widgets.ToggleButtons(
        options=['none','magma', 'viridis', 'seismic', 'gist_rainbow', 'gnuplot', 'hsv_r', 'hsv'],
        description='Colormap:',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[],
    )

    cm_values.inter = widgets.ToggleButtons(
        options=['nearest', 'bilinear', 'bicubic', 'gaussian'],
        description='Interpolation:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[],
    )

    layout = {'width':'90%', 'height': '50px', 'border': 'solid', 'fontcolor':'lightgreen'}
    style_green = {'handle_color': 'green', 'readout_color': 'red', 'slider_color': 'blue'}
    cm_values.f=widgets.FloatSlider(min=0,max=2,step=0.1,value=0.4, continuous_update=False, layout=layout, style=style_green, description="Alpha")

    display(cm_values.color, cm_values.inter, cm_values.f)

    confusion()

def dash():
    print('>> 1) Specify data path (folder selection opens in new window)' '\n')
    print('>> 2) Select .pth file to load (usually in models Folder)')
    path_sp = widgets.Button(description='Specify Path Folder')

    #specify path
    display(path_sp)
    out = widgets.Output()
    display(out)

    def on_path_button(b):
        with out:
            clear_output()
            path_choice_two()
            arch_choice()
    path_sp.on_click(on_path_button)

####################################
## Modules for TN, FN, TP, FP tab ##
####################################
def TN(heatmap=None, heatmap_thresh=16, figsize=(5,5), return_fig=None):
    heatmap=heatmap_choice.choice

    if heatmap==True:
        path = Path(path_choice_two.path)
        dirName = (path/'True_Negative')

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Saving in: " , dirName ,  " Created ")
        else:
            print("Saving in: " , dirName ,  " already exists \n")

        print('Saving images....\n')

        for i, idx in enumerate(inference.tl_idx):
            da, cl = inference.interp.data.dl(inference.interp.ds_type).dataset[idx]
            cl = int(cl)
            preds = inference.interp.preds
            classes = inference.interp.data.classes
            pred_class = inference.interp.pred_class
            pred_class_np2 = (pred_class).data.cpu().numpy()[idx]
            losses = inference.interp.losses
            fn = inference.interp.data.valid_ds.x.items[idx]

            if cl == pred_class_np2 & pred_class_np2 == 0:
                im = show_image(da)
                im.set_title(f' Index: {idx}, Actual_Value: {cl}, Actual_Class: {classes[cl]}\n Pred_Class: {pred_class_np2}, Pred_Class: {classes[pred_class[idx]]}\n\n Preds: {preds[idx][1]}, Loss: {losses[idx]}\n\n File Loc: {fn}')

                figname = (f'TN_{idx}_tlabel_{classes[cl]}_plabel_{classes[pred_class[idx]]}.jpeg'.format(i))
                dest = os.path.join(dirName, figname)
                plt.savefig(dest)

        print(f'Images saved in directory: {dirName}')

    if heatmap==False:
        path = Path(path_choice_two.path)
        fig,axes = plt.subplots(figsize=figsize)
        dirName = (path/'True_Negative_heatmap')

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Saving in: " , dirName ,  " Created ")
        else:
            print("Saving in: " , dirName ,  " already exists \n")

        print(f'Colormap: {cm_values.color.value}')
        print(f'Interpolation: {cm_values.inter.value}')
        print(f'Alpha: {cm_values.f.value} \n')

        print('Saving images....\n')

        for i, idx in enumerate(inference.tl_idx):
            im, cl = inference.interp.data.dl(inference.interp.ds_type).dataset[idx]
            cl = int(cl)
            preds = inference.interp.preds
            classes = inference.interp.data.classes
            pred_class = inference.interp.pred_class
            pred_class_np2 = (pred_class).data.cpu().numpy()[idx]
            losses = inference.interp.losses
            c = np.round(losses.numpy(), decimals=1)
            fn = inference.interp.data.valid_ds.x.items[idx]

            if cl == pred_class_np2 & pred_class_np2 == 0:
                im.show(ax=axes, title=
                    f'Index: {idx}, Actual_Value: {cl}, Actual_Class: {classes[cl]}\n Pred_Class: {pred_class_np2}, Pred_Class: {classes[pred_class[idx]]}\n\n Preds: {preds[idx][1]}, Loss: {losses[idx]}\n\n File Loc: {fn}')

                xb,_=inference.interp.data.one_item(im, detach=False, denorm=False)
                m = inference.interp.learn.model.eval()
                with hook_output(m[0]) as hook_a:
                    with hook_output(m[0], grad=True) as hook_g:
                        preds = m(xb)
                        preds[0,cl].backward()
                acts = hook_a.stored[0].cpu()
                if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
                    grad = hook_g.stored[0][0].cpu()
                    grad_chan = grad.mean(1).mean(1)
                    mult = F.relu(((acts*grad_chan[...,None,None])).sum(0))
                    sz = list(im.shape[-2:])
                    axes.imshow(mult, alpha=cm_values.f.value, extent=(0,*sz[::-1],0), interpolation=cm_values.inter.value, cmap=cm_values.color.value)
                    figname = (f'TN_heat{idx}_tlabel_{classes[cl]}_plabel_{classes[pred_class[idx]]}.jpeg'.format(i))
                    dest = os.path.join(dirName, figname)
                    plt.savefig(dest)
            if ifnone(return_fig, defaults.return_fig): return fig
        print(f'Colormap images saved in directory: {dirName}')

def TP(heatmap=None, heatmap_thresh=16, figsize=(5,5), return_fig=None):
    heatmap=heatmap_choice.choice

    if heatmap==True:
        path = Path(path_choice_two.path)
        dirName = (path/'True_Positive')

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Saving in: " , dirName ,  " Created ")
        else:
            print("Saving in: " , dirName ,  " already exists")

        print('Saving images....\n')

        for i, idx in enumerate(inference.tl_idx):
            da, cl = inference.interp.data.dl(inference.interp.ds_type).dataset[idx]
            cl = int(cl)
            preds = inference.interp.preds
            classes = inference.interp.data.classes
            pred_class = inference.interp.pred_class
            pred_class_np2 = (pred_class).data.cpu().numpy()[idx]
            losses = inference.interp.losses
            fn = inference.interp.data.valid_ds.x.items[idx]

            if cl == pred_class_np2 & pred_class_np2 == 1:
                im = show_image(da)
                im.set_title(f' Index: {idx}, Actual_Value: {cl}, Actual_Class: {classes[cl]}\n Pred_Class: {pred_class_np2}, Pred_Class: {classes[pred_class[idx]]}\n\n Preds: {preds[idx][1]}, Loss: {losses[idx]}\n\n File Loc: {fn}')

                figname = (f'TP_{idx}_tlabel_{classes[cl]}_plabel_{classes[pred_class[idx]]}.jpeg'.format(i))
                dest = os.path.join(dirName, figname)
                plt.savefig(dest)

        print(f'Images saved in directory: {dirName}')

    if heatmap==False:
        path = Path(path_choice_two.path)
        fig,axes = plt.subplots(figsize=figsize)
        dirName = (path/'True_Positive_heatmap')

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Saving in: " , dirName ,  " Created ")
        else:
            print("Saving in: " , dirName ,  " already exists \n")

        print(f'Colormap: {cm_values.color.value}')
        print(f'Interpolation: {cm_values.inter.value}')
        print(f'Alpha: {cm_values.f.value} \n')

        print('Saving images....\n')

        for i, idx in enumerate(inference.tl_idx):
            im, cl = inference.interp.data.dl(inference.interp.ds_type).dataset[idx]
            cl = int(cl)
            preds = inference.interp.preds
            classes = inference.interp.data.classes
            pred_class = inference.interp.pred_class
            pred_class_np2 = (pred_class).data.cpu().numpy()[idx]
            losses = inference.interp.losses
            c = np.round(losses.numpy(), decimals=1)
            fn = inference.interp.data.valid_ds.x.items[idx]

            if cl == pred_class_np2 & pred_class_np2 == 1:
                im.show(ax=axes, title=
                    f'Index: {idx}, Actual_Value: {cl}, Actual_Class: {classes[cl]}\n Pred_Class: {pred_class_np2}, Pred_Class: {classes[pred_class[idx]]}\n\n Preds: {preds[idx][1]}, Loss: {losses[idx]}\n\n File Loc: {fn}')

                xb,_=inference.interp.data.one_item(im, detach=False, denorm=False)
                m = inference.interp.learn.model.eval()
                with hook_output(m[0]) as hook_a:
                    with hook_output(m[0], grad=True) as hook_g:
                        preds = m(xb)
                        preds[0,cl].backward()
                acts = hook_a.stored[0].cpu()
                if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
                    grad = hook_g.stored[0][0].cpu()
                    grad_chan = grad.mean(1).mean(1)
                    mult = F.relu(((acts*grad_chan[...,None,None])).sum(0))
                    sz = list(im.shape[-2:])
                    axes.imshow(mult, alpha=cm_values.f.value, extent=(0,*sz[::-1],0), interpolation=cm_values.inter.value, cmap=cm_values.color.value)
                    figname = (f'TP_heat{idx}_tlabel_{classes[cl]}_plabel_{classes[pred_class[idx]]}.jpeg'.format(i))
                    dest = os.path.join(dirName, figname)
                    plt.savefig(dest)
            if ifnone(return_fig, defaults.return_fig): return fig
        print(f'Colormap images saved in directory: {dirName}')

def FP(heatmap=None, heatmap_thresh=16, figsize=(5,5), return_fig=None):
    heatmap=heatmap_choice.choice

    if heatmap==True:
        path = Path(path_choice_two.path)
        dirName = (path/'False_Positive')

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Saving in: " , dirName ,  " Created ")
        else:
            print("Saving in: " , dirName ,  " already exists")

        print('Saving images....\n')

        for i, idx in enumerate(inference.tl_idx):
            da, cl = inference.interp.data.dl(inference.interp.ds_type).dataset[idx]
            cl = int(cl)
            preds = inference.interp.preds
            classes = inference.interp.data.classes
            pred_class = inference.interp.pred_class
            pred_class_np2 = (pred_class).data.cpu().numpy()[idx]
            losses = inference.interp.losses
            fn = inference.interp.data.valid_ds.x.items[idx]

            if cl != pred_class_np2 & pred_class_np2 == 1:
                im = show_image(da)
                im.set_title(f' Index: {idx}, Actual_Value: {cl}, Actual_Class: {classes[cl]}\n Pred_Class: {pred_class_np2}, Pred_Class: {classes[pred_class[idx]]}\n\n Preds: {preds[idx][1]}, Loss: {losses[idx]}\n\n File Loc: {fn}')

                figname = (f'FP_{idx}_tlabel_{classes[cl]}_plabel_{classes[pred_class[idx]]}.jpeg'.format(i))
                dest = os.path.join(dirName, figname)
                plt.savefig(dest)

        print(f'Images saved in directory: {dirName}')

    if heatmap==False:
        path = Path(path_choice_two.path)
        fig,axes = plt.subplots(figsize=figsize)
        dirName = (path/'False_Positive_heatmap')

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Saving in: " , dirName ,  " Created ")
        else:
            print("Saving in: " , dirName ,  " already exists \n")

        print(f'Colormap: {cm_values.color.value}')
        print(f'Interpolation: {cm_values.inter.value}')
        print(f'Alpha: {cm_values.f.value} \n')

        print('Saving images....\n')

        for i, idx in enumerate(inference.tl_idx):
            im, cl = inference.interp.data.dl(inference.interp.ds_type).dataset[idx]
            cl = int(cl)
            preds = inference.interp.preds
            classes = inference.interp.data.classes
            pred_class = inference.interp.pred_class
            pred_class_np2 = (pred_class).data.cpu().numpy()[idx]
            losses = inference.interp.losses
            c = np.round(losses.numpy(), decimals=1)
            fn = inference.interp.data.valid_ds.x.items[idx]

            if cl != pred_class_np2 & pred_class_np2 == 1:
                im.show(ax=axes, title=
                    f'Index: {idx}, Actual_Value: {cl}, Actual_Class: {classes[cl]}\n Pred_Class: {pred_class_np2}, Pred_Class: {classes[pred_class[idx]]}\n\n Preds: {preds[idx][1]}, Loss: {losses[idx]}\n\n File Loc: {fn}')

                xb,_=inference.interp.data.one_item(im, detach=False, denorm=False)
                m = inference.interp.learn.model.eval()
                with hook_output(m[0]) as hook_a:
                    with hook_output(m[0], grad=True) as hook_g:
                        preds = m(xb)
                        preds[0,cl].backward()
                acts = hook_a.stored[0].cpu()
                if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
                    grad = hook_g.stored[0][0].cpu()
                    grad_chan = grad.mean(1).mean(1)
                    mult = F.relu(((acts*grad_chan[...,None,None])).sum(0))
                    sz = list(im.shape[-2:])
                    axes.imshow(mult, alpha=cm_values.f.value, extent=(0,*sz[::-1],0), interpolation=cm_values.inter.value, cmap=cm_values.color.value)
                    figname = (f'FP_heat{idx}_tlabel_{classes[cl]}_plabel_{classes[pred_class[idx]]}.jpeg'.format(i))
                    dest = os.path.join(dirName, figname)
                    plt.savefig(dest)
            if ifnone(return_fig, defaults.return_fig): return fig
        print(f'Colormap images saved in directory: {dirName}')

def FN(heatmap=None, heatmap_thresh=16, figsize=(5,5), return_fig=None):
    heatmap=heatmap_choice.choice

    if heatmap==True:
        path = Path(path_choice_two.path)
        dirName = (path/'False_Negative')

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Saving in: " , dirName ,  " Created ")
        else:
            print("Saving in: " , dirName ,  " already exists")

        print('Saving images....\n')

        for i, idx in enumerate(inference.tl_idx):
            da, cl = inference.interp.data.dl(inference.interp.ds_type).dataset[idx]
            cl = int(cl)
            preds = inference.interp.preds
            classes = inference.interp.data.classes
            pred_class = inference.interp.pred_class
            pred_class_np2 = (pred_class).data.cpu().numpy()[idx]
            losses = inference.interp.losses
            fn = inference.interp.data.valid_ds.x.items[idx]

            if cl != pred_class_np2 & pred_class_np2 == 0:
                im = show_image(da)
                im.set_title(f' Index: {idx}, Actual_Value: {cl}, Actual_Class: {classes[cl]}\n Pred_Class: {pred_class_np2}, Pred_Class: {classes[pred_class[idx]]}\n\n Preds: {preds[idx][1]}, Loss: {losses[idx]}\n\n File Loc: {fn}')

                figname = (f'TP_{idx}_tlabel_{classes[cl]}_plabel_{classes[pred_class[idx]]}.jpeg'.format(i))
                dest = os.path.join(dirName, figname)
                plt.savefig(dest)

        print(f'Images saved in directory: {dirName}')

    if heatmap==False:
        path = Path(path_choice_two.path)
        fig,axes = plt.subplots(figsize=figsize)
        dirName = (path/'False_Negative_heatmap')

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Saving in: " , dirName ,  " Created ")
        else:
            print("Saving in: " , dirName ,  " already exists \n")

        print(f'Colormap: {cm_values.color.value}')
        print(f'Interpolation: {cm_values.inter.value}')
        print(f'Alpha: {cm_values.f.value} \n')

        print('Saving images....\n')

        for i, idx in enumerate(inference.tl_idx):
            im, cl = inference.interp.data.dl(inference.interp.ds_type).dataset[idx]
            cl = int(cl)
            preds = inference.interp.preds
            classes = inference.interp.data.classes
            pred_class = inference.interp.pred_class
            pred_class_np2 = (pred_class).data.cpu().numpy()[idx]
            losses = inference.interp.losses
            c = np.round(losses.numpy(), decimals=1)
            fn = inference.interp.data.valid_ds.x.items[idx]

            if cl != pred_class_np2 & pred_class_np2 == 0:
                im.show(ax=axes, title=
                    f'Index: {idx}, Actual_Value: {cl}, Actual_Class: {classes[cl]}\n Pred_Class: {pred_class_np2}, Pred_Class: {classes[pred_class[idx]]}\n\n Preds: {preds[idx][1]}, Loss: {losses[idx]}\n\n File Loc: {fn}')

                xb,_=inference.interp.data.one_item(im, detach=False, denorm=False)
                m = inference.interp.learn.model.eval()
                with hook_output(m[0]) as hook_a:
                    with hook_output(m[0], grad=True) as hook_g:
                        preds = m(xb)
                        preds[0,cl].backward()
                acts = hook_a.stored[0].cpu()
                if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
                    grad = hook_g.stored[0][0].cpu()
                    grad_chan = grad.mean(1).mean(1)
                    mult = F.relu(((acts*grad_chan[...,None,None])).sum(0))
                    sz = list(im.shape[-2:])
                    axes.imshow(mult, alpha=cm_values.f.value, extent=(0,*sz[::-1],0), interpolation=cm_values.inter.value, cmap=cm_values.color.value)
                    figname = (f'FN_heat{idx}_tlabel_{classes[cl]}_plabel_{classes[pred_class[idx]]}.jpeg'.format(i))
                    dest = os.path.join(dirName, figname)
                    plt.savefig(dest)
            if ifnone(return_fig, defaults.return_fig): return fig
        print(f'Colormap images saved in directory: {dirName}')

##########################
## Modules for Code tab ##
##########################
def view_code():
    """Helper for write_code"""
    button = widgets.Button(description='View Code')

    display(button)

    out_vc = widgets.Output()
    display(out_vc)

    def on_choice_button(b):
        with out_vc:
            clear_output()
            write_code()

    button.on_click(on_choice_button)

def write_code():
    cut_outs = dashboard_two.cut.value
    jits = dashboard_two.jit.value
    contrasts = dashboard_two.contrast.value
    brights = dashboard_two.bright.value
    rotates = dashboard_two.rotate.value
    syms = dashboard_two.sym_warp.value

    """Helper for code writing"""

    flip = ''
    vert = ''

    if dashboard_two.doflip.value == 'True':
        flip = 'True'
    else:
        flip = 'False'

    if dashboard_two.dovert.value == 'True':
        vert = 'True'
    else:
        vert = 'False'
    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size
    flip_val = dashboard_two.doflip.value #to use for print statements only
    vert_val = dashboard_two.dovert.value #to use for print statements only
    max_zoom=dashboard_two.three.value
    max_warp=dashboard_two.seven.value
    max_light=dashboard_two.five.value
    p_affine=dashboard_two.four.value
    p_light=dashboard_two.six.value
    cut1 = int(cutout_choice.one.value)
    cut2 = int(cutout_choice.two.value)
    length = int(cutout_choice.three.value)
    height = int(cutout_choice.four.value)
    jit1 = float(jit_choice.one.value)
    jit2 = float(jit_choice.two.value)
    scale1 = float(contrast_choice.one.value)
    scale2 = float(contrast_choice.two.value)
    b1 = float(bright_choice.one.value)
    b2 = float(bright_choice.two.value)
    b3 = float(bright_choice.three.value)
    degree1 = float(rotate_choice.one.value)
    degree2 = float(rotate_choice.two.value)
    deg_p = float(rotate_choice.three.value)
    s_war = float(sym_w.one.value)
    s_war2 = float(sym_w.two.value)
    p_sym = float(sym_w.three.value)
    cycle_l = int(training_ds.cl.value)
    path = path_choice.path
    data_code=''

    a, b = metrics_list(mets_list, mets_list_code)
    b_ = b[0]
    bs = str(b_)

    if cut_outs == 'True':
        code_line =(f'cutout(n_holes=({cut1}, {cut2}), length=({length}, {height}), p=1.),')
    else:
        code_line=""
    if jits == 'True':
        code_line_2 =(f'jitter(magnitude={jit1}, p={jit2}),')
    else:
        code_line_2=('')
    if contrasts == 'True':
        code_line_3 = (f'contrast(scale=({scale1}, {scale2}), p=1.),')
    else:
        code_line_3 =('')
    if brights == 'True':
        code_line_4 = (f'brightness(change=({b1}, {b2}), p={b3}),')
    else:
        code_line_4=('')
    if rotates == 'True':
        code_line_5 = (f'rotate(degrees=({degree1}, {degree2}), p={deg_p}),')
    else:
        code_line_5=('')
    if syms == 'True':
        code_line_6 = (f'symmetric_warp(magnitude=({s_war}, {s_war2}), p={p_sym})')
    else:
        code_line_6 =('')

    print(f'from fastai.vision import* \nfrom fastai.widgets import * \nfrom fastai.callbacks import* \nfrom fastai.widgets import ClassConfusion\n')
    print(f"path = '{path}'\n")
    print(f'xtra_tfms=[{code_line} {code_line_2} {code_line_3} {code_line_4} {code_line_5} {code_line_6}]')
    print(f'\ntfms = get_tranforms(do_flip={flip}, flip_vert={vert}, max_zoom={max_zoom},'
          f'max_lighting={max_light}, max_warp={max_warp}, p_affine={p_affine}, p_lighting={p_light}, xtra_tfms=xtra_tfms)\n')
    get_data()
    print(f'data.normalize({dashboard_one.norma.value})\n')
    print(f'data.show_batch(rows=4, figsize=(10,10))\n')

    print(f'learn = cnn_learner(data, base_arch={arch_work.code_i}, pretrained={dashboard_one.pretrain_check.value}, metrics={b},'
                                f'custom_head=None)\n')
    print(f'learn.lr_find()')
    print(f'learn.recorder.plot()\n')

    #values for saving model
    print(f'value_mone = str({dashboard_one.archi.value})')
    print(f'value_mtwo = str({dashboard_one.pretrain_check.value})')
    print(f'value_mthree = str({(round(dashboard_one.f.value))})')
    print(f'value_mfour = str({(round(dashboard_one.m.value))})\n')
    print(f"file_model_name = value_mone + '_pretrained_' + value_mtwo + '_batch_' + value_mthree + '_image_' + value_mfour")

    print(f"learn.fit_one_cycle({cycle_l}, slice({lr_work.info}), "
          f"callbacks=[SaveModelCallback(learn, every='improvement', monitor='{bs}', name='best_' + file_model_name)]\n")
    print(f'learn.save(file_model_name)')
    print(f'preds,y,losses = learn.get_preds(with_loss=True)')
    print(f'interp = ClassificationInterpretation(learn, preds, y, losses)')
    print(f'losses, ids = interp.top_losses(interp.data.c)\n')
    print(f'_, tl_idx = inference.interp.top_losses(len(inference.interp.losses))\n')

    print('>> Confusion Matrix')
    print('class_num = data.c')
    print('if class_num > 2:')
    print('interp.plot_confusion_matrix(figsize=(5,5))')

def display_ui():
    """ Display tabs for visual display"""
    button = widgets.Button(description="Train")
    button_b = widgets.Button(description="Metrics")
    button_m = widgets.Button(description='Model')
    button_l = widgets.Button(description='LR')

    test_button = widgets.Button(description='Batch')
    test2_button = widgets.Button(description='Test2')

    out1a = widgets.Output()
    out1 = widgets.Output()
    out2 = widgets.Output()
    out3 = widgets.Output()
    out4 = widgets.Output()
    out5 = widgets.Output()
    out6 = widgets.Output()
    out7 = widgets.Output()
    out8 = widgets.Output()

    data1a = pd.DataFrame(np.random.normal(size = 50))
    data1 = pd.DataFrame(np.random.normal(size = 100))
    data2 = pd.DataFrame(np.random.normal(size = 150))
    data3 = pd.DataFrame(np.random.normal(size = 200))
    data4 = pd.DataFrame(np.random.normal(size = 250))
    data5 = pd.DataFrame(np.random.normal(size = 300))
    data6 = pd.DataFrame(np.random.normal(size = 350))
    data7 = pd.DataFrame(np.random.normal(size= 400))
    data8 = pd.DataFrame(np.random.normal(size=450))

    with out1a: #info
        clear_output()
        dashboard_one()

    with out1: #data
        clear_output()
        print('>> Choose data source')
        ds()

    with out2: #augmentation
        clear_output()
        print('>> Choose Augmentation Image first:')
        dashboard_two()

    with out3: #Batch
        clear_output()
        print('>> Press button to view batch')
        display(test_button)
        out_batch = widgets.Output()
        display(out_batch)
        def on_choice_button(b):
            with out_batch:
                clear_output()
                batch_val = int(dashboard_one.f.value) # batch size
                image_val = int(dashboard_one.m.value) # image size
                flip_val = dashboard_two.doflip.value #to use for print statements only
                vert_val = dashboard_two.dovert.value #to use for print statements only

                max_zoom=dashboard_two.three.value
                max_warp=dashboard_two.seven.value
                max_light=dashboard_two.five.value
                p_affine=dashboard_two.four.value
                p_light=dashboard_two.six.value
                cut1 = int(cutout_choice.one.value)
                cut2 = int(cutout_choice.two.value)
                length = int(cutout_choice.three.value)
                height = int(cutout_choice.four.value)
                jit1 = float(jit_choice.one.value)
                jit2 = float(jit_choice.two.value)
                scale1 = float(contrast_choice.one.value)
                scale2 = float(contrast_choice.two.value)
                b1 = float(bright_choice.one.value)
                b2 = float(bright_choice.two.value)
                b3 = float(bright_choice.three.value)
                degree1 = float(rotate_choice.one.value)
                degree2 = float(rotate_choice.two.value)
                deg_p = float(rotate_choice.three.value)
                s_war = float(sym_w.one.value)
                s_war2 = float(sym_w.two.value)
                p_sym = float(sym_w.three.value)

                print('>> Training Parameters')
                print(f'Normalization: {dashboard_one.norma.value}| Architecture: {dashboard_one.archi.value}| Pretrain: {dashboard_one.pretrain_check.value}|'
                f'Batch Size: {dashboard_one.f.value}| Image Size: {dashboard_one.m.value}')

                print('\n>> Augmentations')
                print(f'do_flip:  {flip_val}| flip_vert: {vert_val}| max_zoom: {max_zoom}| max_warp: {max_warp}| '
                f'p_affine: {p_affine}| max_lighting: {max_light}| p_lighthing: {p_light}')

                print('\n>> Additional Augmentations')
                print(f'cutout(n_holes=({cut1}, {cut2}, length=({length}, {height}), p=1.), jitter(magnitude={jit1},'
                f'p={jit2}), contrast(scale=({scale1}, {scale2}, p=1.), brightness(change=({b1}, {b2}), p={b3}),'
                f'rotate(degrees({degree1}, {degree2}), p={deg_p}))')
                print(f'\n Padding: {dashboard_two.pad.value}')
                get_data()

        test_button.on_click(on_choice_button)

    with out4: #model
        button_mo = widgets.Button(description='Model')
        display(button_mo)

        out_mod = widgets.Output()
        display(out_mod)

        def on_button_clicked(b):
            with out_mod:
                clear_output()
                print('>> View Model information (model_summary, model[0], model[1]')
                model_summary()

        button_mo.on_click(on_button_clicked)

    with out5: #Metrics
        print ('>> Click button to choose appropriate metrics')
        button_m = widgets.Button(description='Metrics')
        display(button_m)

        out = widgets.Output()
        display(out)

        def on_button_clicked_learn(b):
            with out:
                clear_output()
                arch_work()
                metrics_dashboard()
                #get_data()

        button_m.on_click(on_button_clicked_learn)

    with out6: #train
        button_tr = widgets.Button(description='Train')
        display(button_tr)
        print ('>> Click to view training parameters and learning rate''\n''\n')
        out_tr = widgets.Output()
        display(out_tr)
        def on_button_clicked(b):
            with out_tr:
                clear_output()
                get_data()
        button_tr.on_click(on_button_clicked)

    with out7: #results
        dash()

    with out8: #view code
        print('view code')
        view_code()

    display_ui.tab = widgets.Tab(children = [out1a, out1, out2, out3, out4, out5, out6, out7, out8])
    display_ui.tab.set_title(0, 'Arch')
    display_ui.tab.set_title(1, 'Data')
    display_ui.tab.set_title(2, 'Augmentation')
    display_ui.tab.set_title(3, 'Batch')
    display_ui.tab.set_title(4, 'Model')
    display_ui.tab.set_title(5, 'Metrics')
    display_ui.tab.set_title(6, 'Train')
    display_ui.tab.set_title(7, 'Results')
    display_ui.tab.set_title(8, 'Code')
    display(display_ui.tab)
