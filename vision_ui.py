"""
Vision_UI
Visual graphical interface for Fastai

Last Update: 11/12/2019
https://github.com/asvcode/Vision_UI
"""
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets
import ipywidgets as widgets
from IPython.display import display,clear_output

import pandas as pd

from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import*

import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askdirectory

import webbrowser
from IPython.display import YouTubeVideo

import warnings
warnings.filterwarnings('ignore')

import xresnet2 # for xresnet models

def get_image(image_path):
    print(image_path)

def path_choice():
    root = Tk()
    path_choice.path = askdirectory(title='Select Folder')
    root.destroy()
    print('Folder choice:', {path_choice.path})
    return path_choice.path

def image_choice():
    root = Tk()
    image_choice.path = filedialog.askopenfilename(title='Choose Image')
    root.destroy()
    return image_choice.path

def dashboard_one():
    style = {'description_width': 'initial'}

    print('>> IMPORTANT: Choose folder path under INFO tab if not already choosen' '\n\n' '>> Currently only works with files FROM_FOLDERS' '\n' '>> FROM_DF and FROM_CSV under construction')
    dashboard_one.datain = widgets.ToggleButtons(
        options=['from_folder', 'from_df', 'from_csv'],
        description='Data In:',
        disabled=True,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Data in folder', 'Data in csv format - NOT ACTIVE', 'Data in dataframe - NOT ACTIVE'],
    )
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

    xres_text = widgets.Button(
        description='FOR Xresnet models:  Are not pretrained so have to UNCHECK Pretrain box to avoid errors.',
        disabled=True,
        display='flex',
        flex_flow='column',
        align_items='stretch',
        layout = layout
)
    dashboard_one.pretrain_check = widgets.Checkbox(
        options=['Yes', "No"],
        description='Pretrained:',
        disabled=False,
        value=True,
        box_style='success',
        button_style='lightgreen', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Default: Checked = use pretrained weights, Unchecked = No pretrained weights'],
    )

    layout = {'width':'90%', 'height': '50px', 'border': 'solid', 'fontcolor':'lightgreen'}
    layout_two = {'width':'100%', 'height': '200px', 'border': 'solid', 'fontcolor':'lightgreen'}
    style_green = {'handle_color': 'green', 'readout_color': 'red', 'slider_color': 'blue'}
    style_blue = {'handle_color': 'blue', 'readout_color': 'red', 'slider_color': 'blue'}
    dashboard_one.f=widgets.FloatSlider(min=8,max=64,step=8,value=32, continuous_update=False, layout=layout, style=style_green, description="Batch size")
    dashboard_one.m=widgets.FloatSlider(min=0, max=360, step=16, value=128, continuous_update=False, layout=layout, style=style_green, description='Image size')


    display(dashboard_one.datain, dashboard_one.norma, dashboard_one.archi, xres_text, dashboard_one.pretrain_check, dashboard_one.f, dashboard_one.m)

def dashboard_two():
    choice_button = widgets.Button(description='Augmentation Image')
    button = widgets.Button(description="View")
    print ('>> Choose image to view augmentations: (will open a new window)')

    display(choice_button)

    print('Augmentations')

    layout = {'width':'90%', 'height': '50px', 'border': 'solid', 'fontcolor':'lightgreen'}
    layout_two = {'width':'100%', 'height': '200px', 'border': 'solid', 'fontcolor':'lightgreen'}
    style_green = {'handle_color': 'green', 'readout_color': 'red', 'slider_color': 'blue'}
    style_blue = {'handle_color': 'blue', 'readout_color': 'red', 'slider_color': 'blue'}

    dashboard_two.doflip = widgets.ToggleButtons(
        options=['Yes', "No"],
        description='Do Flip:',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )
    dashboard_two.dovert = widgets.ToggleButtons(
        options=['Yes', "No"],
        description='Do Vert:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
    )
    dashboard_two.two = widgets.FloatSlider(min=0,max=20,step=1,value=10, description='Max Rotate', orientation='vertical', style=style_green, layout=layout_two)
    dashboard_two.three = widgets.FloatSlider(min=1.1,max=4,step=1,value=1.1, description='Max Zoom', orientation='vertical', style=style_green, layout=layout_two)
    dashboard_two.four = widgets.FloatSlider(min=0.25, max=1.0, step=0.1, value=0.75, description='p_affine', orientation='vertical', style=style_green, layout=layout_two)
    dashboard_two.five = widgets.FloatSlider(min=0.2,max=0.99, step=0.1,value=0.2, description='Max Lighting', orientation='vertical', style=style_blue, layout=layout_two)
    dashboard_two.six = widgets.FloatSlider(min=0.25, max=1.1, step=0.1, value=0.75, description='p_lighting', orientation='vertical', style=style_blue, layout=layout_two)
    dashboard_two.seven = widgets.FloatSlider(min=0.1, max=0.9, step=0.1, value=0.2, description='Max warp', orientation='vertical', style=style_green, layout=layout_two)

    ui2 = widgets.VBox([dashboard_two.doflip, dashboard_two.dovert])
    ui = widgets.HBox([dashboard_two.two,dashboard_two.three, dashboard_two.seven, dashboard_two.four,dashboard_two.five, dashboard_two.six])
    ui3 = widgets.HBox([ui2, ui])

    display (ui3)

    print ('>> Press button to view augmentations.  Pressing the button again will let you view additional augmentations below')
    display(button)

    def on_choice_button(b):
        image_choice()
    choice_button.on_click(on_choice_button)

    def on_button_clicked(b):
        image_path = image_choice.path
        print('>> Displaying augmetations')
        display_augs(image_path)

    button.on_click(on_button_clicked)

def display_augs(image_path):

    get_image(image_path)
    image_d = open_image(image_path)
    print(image_d)
    def get_ex(): return open_image(image_path)

    out_flip = dashboard_two.doflip.value #do flip
    out_vert = dashboard_two.dovert.value # do vert
    out_rotate = dashboard_two.two.value #max rotate
    out_zoom = dashboard_two.three.value #max_zoom
    out_affine = dashboard_two.four.value #p_affine
    out_lighting = dashboard_two.five.value #Max_lighting
    out_plight = dashboard_two.six.value #p_lighting
    out_warp = dashboard_two.seven.value #Max_warp

    tfms = get_transforms(do_flip=out_flip, flip_vert=out_vert, max_zoom=out_zoom,
                          p_affine=out_affine, max_lighting=out_lighting, p_lighting=out_plight, max_warp=out_warp,
                         max_rotate=out_rotate)

    _, axs = plt.subplots(2,4,figsize=(12,6))
    for ax in axs.flatten():
        img = get_ex().apply_tfms(tfms[0], get_ex(), size=224)
        img.show(ax=ax)

def metrics_dashboard():
    button = widgets.Button(description="Metrics")

    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size

    tfms = get_transforms(do_flip=dashboard_two.doflip.value, flip_vert=dashboard_two.dovert.value, max_zoom=dashboard_two.three.value,
                          p_affine=dashboard_two.four.value, max_lighting=dashboard_two.five.value, p_lighting=dashboard_two.six.value,
                          max_warp=dashboard_two.seven.value, max_rotate=dashboard_two.two.value, xtra_tfms=None)

    path = path_choice.path
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')

    layout = {'width':'90%', 'height': '50px', 'border': 'solid', 'fontcolor':'lightgreen'}
    style_green = {'button_color': 'green','handle_color': 'green', 'readout_color': 'red', 'slider_color': 'blue'}

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
    ui = widgets.HBox([metrics_dashboard.error_choice, metrics_dashboard.accuracy, metrics_dashboard.topk])
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
            print('Training Metrics''\n')
            print('arch:', dashboard_one.archi.value, '\n''pretrain: ', dashboard_one.pretrain_check.value, '\n' ,'Choosen metrics: ',metrics_list(mets_list))

    button.on_click(on_button_clicked)

def arch_work():
    if dashboard_one.archi.value == 'alexnet':
        arch_work.info = models.alexnet
    elif dashboard_one.archi.value == 'BasicBlock':
        arch_work.info = models.BasicBlock
    elif dashboard_one.archi.value == 'densenet121':
        arch_work.info = models.densenet121
    elif dashboard_one.archi.value == 'densenet161':
        arch_work.info = models.densenet161
    elif dashboard_one.archi.value == 'densenet169':
        arch_work.info = models.densenet169
    elif dashboard_one.archi.value == 'densenet201':
        arch_work.info = models.densenet201
    if dashboard_one.archi.value == 'resnet18':
        arch_work.info = models.resnet18
    elif dashboard_one.archi.value == 'resnet34':
        arch_work.info = models.resnet34
    elif dashboard_one.archi.value == 'resnet50':
        arch_work.info = models.resnet50
    elif dashboard_one.archi.value == 'resnet101':
        arch_work.info = models.resnet101
    elif dashboard_one.archi.value == 'resnet152':
        arch_work.info = models.resnet152
    elif dashboard_one.archi.value == 'squeezenet1_0':
        arch_work.info = models.squeezenet1_0
    elif dashboard_one.archi.value == 'squeezenet1_1':
        arch_work.info = models.squeezenet1_1
    elif dashboard_one.archi.value == 'vgg16_bn':
        arch_work.info = models.vgg16_bn
    elif dashboard_one.archi.value == 'vgg19_bn':
        arch_work.info = models.vgg19_bn
    #elif dashboard_one.archi.value == 'wrn_22':
    #    arch_work.info = models.wrn_22
    elif dashboard_one.archi.value == 'xresnet18':
        arch_work.info = xresnet2.xresnet18 #using xresnet2.py
    elif dashboard_one.archi.value == 'xresnet34':
        arch_work.info = xresnet2.xresnet34_2 #using xresnet2.py
    elif dashboard_one.archi.value == 'xresnet50':
        arch_work.info = xresnet2.xresnet50_2 #using xresent2.py
    elif dashboard_one.archi.value == 'xresnet101':
        arch_work.info = xresnet2.xresnet101 #using xresent2.py
    elif dashboard_one.archi.value == 'xresnet152':
        arch_work.info = xresnet2.xresnet152 #using xresnet2.py

    output = arch_work.info
    output
    print(output)

def view_batch_folder():
    print('>> IMPORTANT: Select data folder under INFO tab prior to clicking on batch button to avoid errors')
    button_g = widgets.Button(description="View Batch")
    display(button_g)

    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size

    out = widgets.Output()
    display(out)

    def on_button_click(b):
        with out:
            clear_output()
            print('\n''Augmentations''\n''Do Flip:', dashboard_two.doflip.value,'|''Do Vert:', dashboard_two.dovert.value, '\n'
                  '\n''Max Rotate: ', dashboard_two.two.value,'|''Max Zoom: ', dashboard_two.three.value,'|''Max Warp: ',
                  dashboard_two.seven.value,'|''p affine: ', dashboard_two.four.value, '\n''Max Lighting: ', dashboard_two.five.value,
                  'p lighting: ', dashboard_two.six.value, '\n'
                  '\n''Normalization Value:', dashboard_one.norma.value, '\n''\n''working....')

            tfms = get_transforms(do_flip=dashboard_two.doflip.value, flip_vert=dashboard_two.dovert.value, max_zoom=dashboard_two.three.value,
                                  p_affine=dashboard_two.four.value, max_lighting=dashboard_two.five.value, p_lighting=dashboard_two.six.value,
                                  max_warp=dashboard_two.seven.value, max_rotate=dashboard_two.two.value, xtra_tfms=None)

            path = path_choice.path
            data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')
            data.normalize(stats_info())
            data.show_batch(rows=5, figsize=(10,10))

    button_g.on_click(on_button_click)

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

mets_list = []

precision = Precision()
recall = Recall()

def metrics_list(mets_list):
    mets_error = metrics_dashboard.error_choice.value
    mets_accuracy= metrics_dashboard.accuracy.value
    mets_accuracy_thr = metrics_dashboard.topk.value
    mets_precision = metrics_dashboard.precision.value
    mets_recall = metrics_dashboard.recall.value
    mets_dice = metrics_dashboard.dice.value

    mets_list=[]
    output_acc = accuracy
    output_thresh = top_k_accuracy
    output = error_rate

    if mets_error == 'Yes':
        mets_list.append(error_rate)
    else:
        None
    if mets_accuracy == 'Yes':
        mets_list.append(accuracy)
    else:
        None
    if mets_accuracy_thr == 'Yes':
        mets_list.append(top_k_accuracy)
    else:
        None
    if mets_precision == 'Yes':
        mets_list.append(precision)
    else:
        None
    if mets_recall == 'Yes':
        mets_list.append(recall)
    else:
        None
    if mets_dice == 'Yes':
        mets_list.append(dice)
    else:
        None

    metrics_info = mets_list

    return mets_list

def model_summary():
    print('>> Review Model information: ', dashboard_one.archi.value)

    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size

    button_summary = widgets.Button(description="Model Summary")
    button_model_0 = widgets.Button(description='Model[0]')
    button_model_1 = widgets.Button(description='Model[1]')

    tfms = get_transforms(do_flip=dashboard_two.doflip.value, flip_vert=dashboard_two.dovert.value, max_zoom=dashboard_two.three.value,
                          p_affine=dashboard_two.four.value, max_lighting=dashboard_two.five.value, p_lighting=dashboard_two.six.value,
                          max_warp=dashboard_two.seven.value, max_rotate=dashboard_two.two.value, xtra_tfms=None)

    path = path_choice.path
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')

    r = dashboard_one.pretrain_check.value

    ui_out = widgets.HBox([button_summary, button_model_0, button_model_1])

    arch_work()

    display(ui_out)
    out = widgets.Output()
    display(out)

    def on_button_clicked_summary(b):
        with out:
            clear_output()
            print('working''\n')
            learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, custom_head=None)
            print('Model Summary')
            info = learn.summary()
            print(info)

    button_summary.on_click(on_button_clicked_summary)

    def on_button_clicked_model_0(b):
        with out:
            clear_output()
            print('working''\n')
            learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, custom_head=None)
            print('Model[0]')
            info_s = learn.model[0]
            print(info_s)

    button_model_0.on_click(on_button_clicked_model_0)

    def on_button_clicked_model_1(b):
        with out:
            clear_output()
            print('working''\n')
            learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, custom_head=None)
            print('Model[1]')
            info_sm = learn.model[1]
            print(info_sm)

    button_model_1.on_click(on_button_clicked_model_1)

def learn_dash():
    button = widgets.Button(description="Learn")
    print ('Choosen metrics: ',metrics_list(mets_list))
    metrics_list(mets_list)

    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size

    r = dashboard_one.pretrain_check.value
    t = metrics_list(mets_list)

    tfms = get_transforms(do_flip=dashboard_two.doflip.value, flip_vert=dashboard_two.dovert.value, max_zoom=dashboard_two.three.value,
                          p_affine=dashboard_two.four.value, max_lighting=dashboard_two.five.value, p_lighting=dashboard_two.six.value,
                          max_warp=dashboard_two.seven.value, max_rotate=dashboard_two.two.value, xtra_tfms=None)

    path = path_choice.path
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')

    learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=metrics_list(mets_list), custom_head=None)

    learn.lr_find()
    learn.recorder.plot()

def ds():
    style = {'description_width': 'initial'}
    ds.datas = widgets.ToggleButtons(
        options=['Folder', 'CATS&DOGS', 'IMAGENETTE',
                 'IMAGENETTE_160', 'IMAGENETTE_320', 'IMAGEWOOF', 'IMAGEWOOF_160', 'IMAGEWOOF_320',
                 'CIFAR', 'CIFAR_100', 'MNIST', 'MNIST_SAMPLE', 'MNIST_TINY'],
        description='Choose',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Choose your folder', ' Cats&Dogs: 25000 images, 819MB',
                  'Imagenette: A subset of 10 easily classified classes from Imagenet, 18000 images, 1.48GB', 'Imagenette_160: 18000 images, 127MB',
                  'Imagenette_320: 18000 images, 358MB', 'ImageWoof: A subset of 10 harder to classify classes from Imagenet, 18000 images, 1.28GB',
                  'ImageWoof_160: 18000 images, 119MB', 'ImageWoof_320: 18000 images, 343MB', 'Cifar: 60000 images, 234MB',
                  'Cifar_100: 100 classes, 60000 images, 234MB', 'Mnist', 'Mnist 14434 images', 'Mnist 1428 images'],
        style=style
    )
    display(ds.datas)

def ds_choice():
    if ds.datas.value == 'Folder':
        path_choice()
    elif ds.datas.value == 'CATS&DOGS':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.DOGS)
    elif ds.datas.value == 'IMAGENETTE':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.IMAGENETTE)
    elif ds.datas.value == 'IMAGENETTE_160':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.IMAGENETTE_160)
    elif ds.datas.value == 'IMAGENETTE_320':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.IMAGENETTE_320)
    elif ds.datas.value == 'IMAGEWOOF':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.IMAGEWOOF)
    elif ds.datas.value == 'IMAGEWOOF_160':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.IMAGEWOOF_160)
    elif ds.datas.value == 'IMAGEWOOF_320':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.IMAGEWOOF_320)
    elif ds.datas.value == 'CIFAR':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.CIFAR)
    elif ds.datas.value == 'CIFAR_100':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.CIFAR_100)
    elif ds.datas.value == 'MNIST':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.MNIST)
    elif ds.datas.value == 'MNIST_SAMPLE':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.MNIST_SAMPLE)
    elif ds.datas.value == 'MNIST_TINY':
        print('downloading dataset')
        path_choice.path = untar_data(URLs.MNIST_TINY)

def version():
    import fastai
    import psutil

    print ('>> Vision_UI Last Update: 11/12/2019')
    style = {'description_width': 'initial'}

    button = widgets.Button(description='System')
    button_one = widgets.Button(description='Choose Folder')
    button_t = widgets.Button(description='Choice')
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

    print ('>> Choose your own dataset (Folder - opens a new window) or a dataset (hover over button to view dataset info) \n\n>> Click Choice button')
    ds()
    display(button_t)

    out_three = widgets.Output()
    display(out_three)

    def on_button_clicked_info2(b):
        with out_three:
            clear_output()
            ds_choice()
            il = ImageList.from_folder(path_choice.path)
            print(f'No of items in folder: {len(il.items)}')
            print('Done!')

    button_t.on_click(on_button_clicked_info2)

    print ('Resources')
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

def info_lr():
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
            print('Data in:', dashboard_one.datain.value,'|' 'Normalization:', dashboard_one.norma.value,'|' 'Architecture:', dashboard_one.archi.value,
                      'Pretrain:', dashboard_one.pretrain_check.value,'\n''Batch Size:', dashboard_one.f.value,'|''Image Size:', dashboard_one.m.value,'\n'
                      '\n''Augmentations''\n''Do Flip:', dashboard_two.doflip.value,'|''Do Vert:', dashboard_two.dovert.value, '\n'
                      '\n''Max Rotate: ', dashboard_two.two.value,'|''Max Zoom: ', dashboard_two.three.value,'|''Max Warp: ',
                      dashboard_two.seven.value,'|''p affine: ', dashboard_two.four.value, '\n''Max Lighting: ', dashboard_two.five.value,
                      'p lighting: ', dashboard_two.six.value, '\n'
                     '\n''Normalization Value:', dashboard_one.norma.value,'\n' '\n''Training Metrics''\n',
                      metrics_list(mets_list))

    button.on_click(on_button_clicked_info)

    def on_button_clicked_info2(b):
        with out:
            clear_output()
            dashboard_one.datain.value, dashboard_one.norma.value, dashboard_one.archi.value, dashboard_one.pretrain_check.value,
            dashboard_one.f.value, dashboard_one.m.value, dashboard_two.doflip.value, dashboard_two.dovert.value,
            dashboard_two.two.value, dashboard_two.three.value, dashboard_two.seven.value, dashboard_two.four.value, dashboard_two.five.value,
            dashboard_two.six.value, dashboard_one.norma.value,metrics_list(mets_list)

            learn_dash()

    button_two.on_click(on_button_clicked_info2)

    def on_button_clicked_info3(b):
        with out:
            clear_output()
            print('Train')
            training()

    button_three.on_click(on_button_clicked_info3)

def lr_work():
    if training.lr.value == '1e-6':
        lr_work.info = float(0.000001)
    elif training.lr.value == '1e-5':
        lr_work.info = float(0.00001)
    elif training.lr.value == '1e-4':
        lr_work.info = float(0.0001)
    elif training.lr.value == '1e-3':
        lr_work.info = float(0.001)
    elif training.lr.value == '1e-2':
        lr_work.info = float(0.01)
    elif training.lr.value == '1e-1':
        lr_work.info = float(0.1)

def training():
    print('>> Using fit_one_cycle')
    button = widgets.Button(description='Train')

    style = {'description_width': 'initial'}

    layout = {'width':'90%', 'height': '50px', 'border': 'solid', 'fontcolor':'lightgreen'}
    layout_two = {'width':'100%', 'height': '200px', 'border': 'solid', 'fontcolor':'lightgreen'}
    style_green = {'handle_color': 'green', 'readout_color': 'red', 'slider_color': 'blue'}
    style_blue = {'handle_color': 'blue', 'readout_color': 'red', 'slider_color': 'blue'}

    training.cl=widgets.FloatSlider(min=1,max=64,step=1,value=1, continuous_update=False, layout=layout, style=style_green, description="Cycle Length")
    training.lr = widgets.ToggleButtons(
        options=['1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1'],
        description='Learning Rate:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        style=style,
        value='1e-2',
        tooltips=['Choose a suitable learning rate'],
    )

    display(training.cl, training.lr)

    display(button)

    out = widgets.Output()
    display(out)

    def on_button_clicked(b):
        with out:
            clear_output()
            lr_work()
            print('>> Training....''\n''Learning Rate: ', lr_work.info)
            dashboard_one.datain.value, dashboard_one.norma.value, dashboard_one.archi.value, dashboard_one.pretrain_check.value,
            dashboard_one.f.value, dashboard_one.m.value, dashboard_two.doflip.value, dashboard_two.dovert.value,
            dashboard_two.two.value, dashboard_two.three.value, dashboard_two.seven.value, dashboard_two.four.value, dashboard_two.five.value,
            dashboard_two.six.value, dashboard_one.norma.value,metrics_list(mets_list)

            metrics_list(mets_list)

            batch_val = int(dashboard_one.f.value) # batch size
            image_val = int(dashboard_one.m.value) # image size

            #values for saving model
            value_mone = str(dashboard_one.archi.value)
            value_mtwo = str(dashboard_one.pretrain_check.value)
            value_mthree = str(round(dashboard_one.f.value))
            value_mfour = str(round(dashboard_one.m.value))

            r = dashboard_one.pretrain_check.value

            tfms = get_transforms(do_flip=dashboard_two.doflip.value, flip_vert=dashboard_two.dovert.value, max_zoom=dashboard_two.three.value,
                          p_affine=dashboard_two.four.value, max_lighting=dashboard_two.five.value, p_lighting=dashboard_two.six.value,
                          max_warp=dashboard_two.seven.value, max_rotate=dashboard_two.two.value, xtra_tfms=None)

            path = path_choice.path
            #data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')

            data = (ImageList.from_folder(path)
                     .split_by_folder()
                     .label_from_folder()
                     .transform(tfms, size=image_val)
                     .add_test_folder('test')
                     .databunch(path=path))

            learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=metrics_list(mets_list), custom_head=None, callback_fns=ShowGraph)

            cycle_l = int(training.cl.value)

            learn.fit_one_cycle(cycle_l, slice(lr_work.info))

            #save model
            file_model_name = value_mone + '_pretrained_' + value_mtwo + '_batch_' + value_mthree + '_image_' + value_mfour

            learn.save(file_model_name)

    button.on_click(on_button_clicked)

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

        data = ImageDataBunch.from_folder(path_choice_two.path, valid='test',
                                       ds_tfms=tfms, bs=128,
                                       size=128)

        loading_model.learn = cnn_learner(data, base_arch=arch_working.info, pretrained=arch_choice.pretrain_check.value, custom_head=None)
        loading_model.learn.load(load_model.model_path_2a)
        print('>> Model loaded')
        print('>> Getting Intepretations....')
        inference()
        print('Done')
        rs()

    loading_button.on_click(on_loading_clicked)

def load_model():
    load_model.model_path_2 = model_choice.path.split('.pth')
    load_model.model_path_2a = load_model.model_path_2[0]
    loading_model()

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

def rs():
    print('>> Model loaded: ', model_choice.path)
    print('>> Use options below to view results')

    plot_button = widgets.Button(description='Multi_Top_Losses')
    cmap_button = widgets.Button(description='Top_Losses')
    cm_button = widgets.Button(description='Confusion Matrix')

    dip = widgets.HBox([cm_button, plot_button, cmap_button])

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

def inference():
    preds,y,losses = loading_model.learn.get_preds(with_loss=True)
    inference.interp = ClassificationInterpretation(loading_model.learn, preds, y, losses)
    losses, ids = inference.interp.top_losses(inference.interp.data.c)

    _, inference.tl_idx = inference.interp.top_losses(len(inference.interp.losses))

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



def display_ui():
    button = widgets.Button(description="Train")
    button_b = widgets.Button(description="Metrics")
    button_m = widgets.Button(description='Model')
    button_l = widgets.Button(description='LR')

    out1a = widgets.Output()
    out1 = widgets.Output()
    out2 = widgets.Output()
    out3 = widgets.Output()
    out4 = widgets.Output()
    out5 = widgets.Output()
    out6 = widgets.Output()
    out7 = widgets.Output()

    data1a = pd.DataFrame(np.random.normal(size = 50))
    data1 = pd.DataFrame(np.random.normal(size = 100))
    data2 = pd.DataFrame(np.random.normal(size = 150))
    data3 = pd.DataFrame(np.random.normal(size = 200))
    data4 = pd.DataFrame(np.random.normal(size = 250))
    data5 = pd.DataFrame(np.random.normal(size = 300))
    data6 = pd.DataFrame(np.random.normal(size = 350))
    data7 = pd.DataFrame(np.random.normal(size= 400))

    with out1a: #info
        version()

    with out1: #data
        dashboard_one()

    with out2: #augmentation
        dashboard_two()

    with out3: #Batch
        print('Click to view Batch' '\n\n')
        view_batch_folder()

    with out4: #model
        print('>> View Model information (model_summary, model[0], model[1])''\n\n''>> For xresnet: Pretrained needs to be set to FALSE, setting to TRUE results in error: NameError: name model_urls is not defined')
        display(button_m)

        out_two = widgets.Output()
        display(out_two)

        def on_button_clicked_train(b):
            with out_two:
                clear_output()
                print('Your pretrained setting: ', dashboard_one.pretrain_check.value)
                model_summary()

    button_m.on_click(on_button_clicked_train)

    with out5: #Metrics
        print ('>> Click button to choose appropriate metrics')
        display(button_b)

        out = widgets.Output()
        display(out)

        def on_button_clicked_learn(b):
            with out:
                clear_output()
                arch_work()
                metrics_dashboard()

    button_b.on_click(on_button_clicked_learn)

    with out6: #train
        print ('>> Click to view training parameters and learning rate''\n''\n'
              '>> IMPORTANT: You have to go through METRICS tab prior to choosing LR')
        info_lr()

    with out7: #results
        dash()

    tab = widgets.Tab(children = [out1a, out1, out2, out3, out4, out5, out6, out7])
    tab.set_title(0, 'Info')
    tab.set_title(1, 'Data')
    tab.set_title(2, 'Augmentation')
    tab.set_title(3, 'Batch')
    tab.set_title(4, 'Model')
    tab.set_title(5, 'Metrics')
    tab.set_title(6, 'Train')
    tab.set_title(7, 'Results')
    display(tab)
