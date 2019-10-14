"""
Paperspace_UI based on Vision_UI
Visual graphical interface for Fastai

Last Update: 10/12/2019
https://github.com/asvcode/Vision_UI
"""

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets
import ipywidgets as widgets
import IPython
from IPython.display import display,clear_output

import webbrowser
from IPython.display import YouTubeVideo

from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import*

def version():
    import fastai
    import psutil

    print ('>> Paperspace version')

    button = widgets.Button(description='System')
    but = widgets.HBox([button])
    display(but)

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

def dashboard_one():
    style = {'description_width': 'initial'}

    print('>> Currently only works with files FROM_FOLDERS' '\n')
    dashboard_one.datain = widgets.ToggleButtons(
        options=['from_folder'],
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
    button = widgets.Button(description="View")
    print ('>> Choose image to view augmentations: (will open a new window)')

    image_choice()
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

    def on_button_clicked(b):
        image_path = str(image_choice.output_variable.value)
        print('>> Displaying augmetations')
        display_augs(image_path)

    button.on_click(on_button_clicked)

def get_image(image_path):
    print(image_path)

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

            path = path_load.path_choice
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

    path = path_load.path_choice
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
        arch_work.info = xresnet2.xresnet18
    elif dashboard_one.archi.value == 'xresnet34':
        arch_work.info = xresnet2.xresnet34
    elif dashboard_one.archi.value == 'xresnet50':
        arch_work.info = xresnet2.xresnet50
    elif dashboard_one.archi.value == 'xresnet101':
        arch_work.info = xresnet2.xresnet101
    elif dashboard_one.archi.value == 'xresnet152':
        arch_work.info = xresnet2.xresnet152

    output = arch_work.info
    output
    print(output)

def metrics_dashboard():
    button = widgets.Button(description="Metrics")

    batch_val = int(dashboard_one.f.value) # batch size
    image_val = int(dashboard_one.m.value) # image size

    tfms = get_transforms(do_flip=dashboard_two.doflip.value, flip_vert=dashboard_two.dovert.value, max_zoom=dashboard_two.three.value,
                          p_affine=dashboard_two.four.value, max_lighting=dashboard_two.five.value, p_lighting=dashboard_two.six.value,
                          max_warp=dashboard_two.seven.value, max_rotate=dashboard_two.two.value, xtra_tfms=None)

    path = path_load.path_choice
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

    path = path_load.path_choice
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')

    learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=metrics_list(mets_list), custom_head=None)

    learn.lr_find()
    learn.recorder.plot()


def model_button():
    button_m = widgets.Button(description='Model')

    print('>> View Model information (model_summary, model[0], model[1])''\n\n''>> For xresnet: Pretrained needs to be set to FALSE')
    display(button_m)

    out_two = widgets.Output()
    display(out_two)

    def on_button_clicked_train(b):
      with out_two:
        clear_output()
        print('Your pretrained setting: ', dashboard_one.pretrain_check.value)
        model_summary()

    button_m.on_click(on_button_clicked_train)

#def drive_upload():
 # from google.colab import drive
  #print('mounting drive')
  #drive.mount('/content/gdrive', force_remount=True)
  #drive_upload.root_dir = "/content/gdrive/My Drive/"
  #print('drive mounted')

def path_load():

  #path = Path(get_path.output_variable)
  file_location = str(get_path.output_variable.value)
  #path_load.path_choice = path/file_location
  path_load.path_choice = file_location

  il = ImageList.from_folder(path_load.path_choice)
  print(len(il.items))
  print(path_load.path_choice)

def image_choice():

  from ipywidgets import widgets
  button_choice = widgets.Button(description="Image Path")

  # Create text widget for output
  image_choice.output_variable = widgets.Text()
  display(image_choice.output_variable)

  display(button_choice)

def get_path():

  from ipywidgets import widgets
  button_choice = widgets.Button(description="Load Path")

  # Create text widget for output
  get_path.output_variable = widgets.Text()
  display(get_path.output_variable)

  display(button_choice)

  def on_button_clicked_summary(b):
      path_load()

  button_choice.on_click(on_button_clicked_summary)

def metric_button():
    button_b = widgets.Button(description="Metrics")
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

            path = path_load.path_choice

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

def display_ui():
    button = widgets.Button(description="Train")
    button_b = widgets.Button(description="Metrics")
    button_m = widgets.Button(description='Model')
    button_l = widgets.Button(description='LR')

    out1aa = widgets.Output()
    out1a = widgets.Output()
    out1 = widgets.Output()
    out2 = widgets.Output()
    out3 = widgets.Output()
    out4 = widgets.Output()
    out5 = widgets.Output()
    out6 = widgets.Output()

    data1aa = pd.DataFrame(np.random.normal(size = 50))
    data1a = pd.DataFrame(np.random.normal(size = 100))
    data1 = pd.DataFrame(np.random.normal(size = 150))
    data2 = pd.DataFrame(np.random.normal(size = 200))
    data3 = pd.DataFrame(np.random.normal(size = 250))
    data4 = pd.DataFrame(np.random.normal(size = 300))
    data5 = pd.DataFrame(np.random.normal(size = 350))
    data6 = pd.DataFrame(np.random.normal(size = 400))

    with out1aa: #path_choice
        print('path')
        get_path()

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

    tab = widgets.Tab(children = [out1aa, out1a, out1, out2, out3, out4, out5, out6])
    tab.set_title(0, 'Path')
    tab.set_title(1, 'Info')
    tab.set_title(2, 'Data')
    tab.set_title(3, 'Augmentation')
    tab.set_title(4, 'Batch')
    tab.set_title(5, 'Model')
    tab.set_title(6, 'Metrics')
    tab.set_title(7, 'Train')
    display(tab)
