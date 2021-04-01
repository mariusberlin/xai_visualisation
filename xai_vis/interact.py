import tensorflow as tf
import tensorflow.keras
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
from matplotlib import pyplot as plt
from IPython.display import clear_output
import xai_vis.methods
import xai_vis.utils

'''
Jupyter notebook visualisation
'''

def vis(model_var,pixel_array):
    
    #if path is provided as model load model
    if type(model) == str:
        model_path = model_var
        
        #load model
        trained_model = tf.keras.models.load_model(model_path, compile=False)
        #get weights
        loaded_weights = trained_model.get_weights()
        #new_model.summary()
        model.set_weights(loaded_weights)
    
    else:
        pass
    
    #transform pixel array from (depth,length,width) to (1,depth,length,width,1)
    if len(np.shape(pixel_array)) == 3:
        pixel_array = np.expand_dims(pixel_array, 3)
        pixel_array = np.expand_dims(pixel_array, 0)

    #call the interpretability methods
    print("Calculating the saliency maps. This may take several minutes.")
    capi_vsali = call_vsaliency(model,model_modifier,loss,pixel_array)
    capi_sali = call_smooth(model,model_modifier,loss,pixel_array)
    capi_grad = call_grad(model,model_modifier,loss,pixel_array)
    capi_gradplus = call_gradplus(model,model_modifier,loss,pixel_array)
    capi_faster_scorecam = call_faster_scorecam(model, model_modifier,loss,pixel_array)


    #clear pritn statement
    clear_output(wait=True)

    #define the widgets
    layer = widgets.IntSlider(description='Slice:', min=0, max=(np.shape(pixel_array)[1] - 1), orientation='horizontal')

    method = widgets.ToggleButtons(
        options=['Vanilla Saliency', 'SmoothGrad', 'GradCam', 'GradCam++', 'ScoreCam'],
        description='Method:',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
        #     icons=['check'] * 3
    )

    attention = widgets.ToggleButtons(
        options=['Slice-wise', "Max"],
        description='Attention:',

        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
        #     icons=['check'] * 3
    )

    alpha = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.001,
        description='Overlay:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    play = widgets.Play(
        value=0,
        min=0,
        max=(np.shape(pixel_array)[1] - 1),
        step=1,
        interval=500,
        description="Press play",
        disabled=False
    )
    widgets.jslink((play, 'value'), (layer, 'value'))
    
    #assemble the widget
    ui = widgets.VBox([attention, method, layer, alpha, play])

    #create the overlay of original image and heatmap
    def explore_3dimage(layer, attention, method, alpha) :
        
        #define the visualize
        if attention == "Slice-wise" :
            attention_mode = 'slice'
        # elif attention == "Average":
        #    attention_mode = 'mean'
        elif attention == "Max" :
            attention_mode = 'max'
        
        #load interpretability method
        if method == 'Vanilla Saliency' :
            heatmap = mode(attention_mode, capi_vsali)
            capi = capi_vsali
        elif method == 'SmoothGrad' :
            heatmap = mode(attention_mode, capi_sali)
            capi = capi_sali
        elif method == 'GradCam' :
            heatmap = mode(attention_mode, capi_grad)
            capi = capi_grad
        elif method == 'GradCam++' :
            heatmap = mode(attention_mode, capi_gradplus)
            capi = capi_gradplus
        elif method == 'ScoreCam' :
            heatmap = mode(attention_mode, capi_faster_scorecam)
            capi = capi_faster_scorecam

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.5])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        mri_image_slice = np.squeeze(pixel_array)[layer, :, :]
        mri_image_slice = np.float64(mri_image_slice)


        #max methods:
        if len(np.shape(heatmap)) == 2 :
            heatmap_slice = heatmap
            
        #slice methods
        elif len(np.shape(heatmap)) == 3 :
            heatmap_slice = np.squeeze(heatmap)[layer, :, :]
            heatmap_slice = np.float64(heatmap_slice)

        ax1.imshow(mri_image_slice)
        ax1.imshow(heatmap_slice, cmap='jet', alpha=alpha)
        ax1.set_title(method + " - " + attention, fontsize=18)
        ax1.axis('off')
        ax2.set_xlabel('Slice', fontsize=13)
        ax2.set_ylabel('Pixel intensities', fontsize=13)
        ax2.set_title("Attention histogram", fontsize=18)

        # calculate GAP in z-direction
        capi = np.squeeze(capi)
        capi_gap = np.apply_over_axes(np.mean, capi, [1, 2])

        # normalize
        capi_gap_norm = (capi_gap - min(capi_gap)) / (max(capi_gap) - min(capi_gap))
        max_slice = np.argmax(capi_gap, axis=0)
        ax2.plot(np.squeeze(capi_gap))
        plt.vlines(x=max_slice, ymin=0, ymax=max(capi_gap), linestyle="--")
        plt.text(max_slice + 0.5, 0.1 * max(capi_gap), "max slice: \n" + str(max_slice[0][0]))
        plt.vlines(x=layer, ymin=0, ymax=np.squeeze(capi_gap)[layer], linestyle="dotted", color="b")
        plt.text(layer + 0.5, 0.03 * max(capi_gap), "current slice: \n" + str(layer))
        plt.grid(axis=('both'), linestyle="--")
        xt = ax2.get_xticks()
        plt.ylim(0)
        ax2.set_xticks(xt)
        ax2.set_xticklabels(xt)
        plt.xlim(left=0, right=(np.shape(pixel_array)[1] - 1))
        fig.subplots_adjust(wspace=0.5)

        return layer

    #interactive output
    out = widgets.interactive_output(explore_3dimage,
                                     {'attention' : attention, 'method' : method, 'layer' : layer, 'alpha' : alpha})
    #ensure smoother sliding through the layers
    out.layout.height = '550px'
    display(ui, out)
