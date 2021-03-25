import matplotlib.gridspec as gridspec

#call the methods

def vis(model_path,pixel_array):

    #load model
    trained_model = tf.keras.models.load_model(model_path, compile=False)
    # get weights
    loaded_weights = trained_model.get_weights()
    # new_model.summary()
    model.set_weights(loaded_weights)


    print("Calculating the saliency maps. This may take several minutes.")
    capi_vsali = call_vsaliency(model,model_modifier,loss,pixel_array))
    capi_sali = call_smooth(loss,pixel_array)
    capi_grad = call_grad(model,model_modifier,loss,pixel_array)
    capi_gradplus = call_gradplus(model,model_modifier,loss,pixel_array)
    capi_faster_scorecam = call_faster_scorecam(model, model_modifier,loss,pixel_array)



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

    ui = widgets.VBox([attention, method, layer, alpha, play])


    def explore_3dimage(layer, attention, method, alpha) :
        # print("Shape heatmap: ",np.shape(heatmap))

        if attention == "Slice-wise" :
            attention_mode = 'slice'
        # elif attention == "Average":
        #    attention_mode = 'mean'
        elif attention == "Max" :
            attention_mode = 'max'

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

        # print(np.shape(heatmap))

        # for mean/max cam only:
        if len(np.shape(heatmap)) == 2 :
            heatmap_slice = heatmap

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
        # ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

        # calculate GAP in z-direction
        capi = np.squeeze(capi)
        capi_gap = np.apply_over_axes(np.mean, capi, [1, 2])

        # normalize
        capi_gap_norm = (capi_gap - min(capi_gap)) / (max(capi_gap) - min(capi_gap))

        max_slice = np.argmax(capi_gap, axis=0)
        ax2.plot(np.squeeze(capi_gap))

        # n, bins, patches = ax2.hist(b, 35, density=False

        # txt = "Histrogram before clipping"
        # fig.text(.5, -.04, txt, ha='center')
        plt.vlines(x=max_slice, ymin=0, ymax=max(capi_gap), linestyle="--")
        plt.text(max_slice + 0.5, 0.1 * max(capi_gap), "max slice: \n" + str(max_slice[0][0]))

        plt.vlines(x=layer, ymin=0, ymax=np.squeeze(capi_gap)[layer], linestyle="dotted", color="b")
        plt.text(layer + 0.5, 0.03 * max(capi_gap), "current slice: \n" + str(layer))

        plt.grid(axis=('both'), linestyle="--")

        xt = ax2.get_xticks()
        # xt=np.append(int(xt),max_slice[0])
        # xtl=xt.tolist()
        # xtl[-1]= max_slice[0]
        plt.ylim(0)

        ax2.set_xticks(xt)
        ax2.set_xticklabels(xt)
        plt.xlim(left=0, right=(np.shape(pixel_array)[1] - 1))
        fig.subplots_adjust(wspace=0.5)

        return layer


    out = widgets.interactive_output(explore_3dimage,
                                     {'attention' : attention, 'method' : method, 'layer' : layer, 'alpha' : alpha})
    out.layout.height = '550px'

    display(ui, out)

