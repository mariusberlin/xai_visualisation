# Find last dense layer
def last_dense_layer(model ):
    for layer in reversed(model.layers ):
        # check to see if the layer has a 4D output
        if str(layer.name[:5]) == "dense ":
            print ("layer dense: ", layer.name)
            return layer.name
            break


# Find last conv layer
def last_conv_layer(model ):
    for layer in reversed(model.layers ):
        # check to see if the layer has a 4D output
        if str(layer.name[:6]) == "conv3d ":
            print ("layer conv: ", layer.name)
            return layer.name
            break


def loss(output ):
    return (output[0][1])  # sample 0 class1

def model_modifier(m ):
    m.layers[-1].activation = tf.keras.activations.linear
    return m

def mode(mode, capi) :
    if mode == "mean" :
        capi = np.squeeze(capi)
        capi = np.mean(capi, axis=0)
        # print("mean capi: ",(capi))
        return capi

    elif mode == "max" :
        capi = np.squeeze(capi)
        capi_gap = np.apply_over_axes(np.mean, capi, [1, 2])
        # print(np.shape(capi_gap))
        # normalize
        # capi_gap = (capi_gap-min(capi_gap))/(max(capi_gap)-min(capi_gap))
        capi_gap_max = np.argmax(capi_gap)
        print(capi_gap_max)

        # print(capi_gap_max)
        return np.squeeze(capi[capi_gap_max, :, :])

    elif mode == "slice" :
        # print("slice")
        capi = np.squeeze(capi)
        return capi


###Masterthesis utils###

def resize(pixel_array) :
    pixel_array = np.clip(pixel_array, None, np.quantile(pixel_array, .99))

    # resizing, 10% left to target shape for cropping
    # t_1 = perf_counter()
    output_shape = (params["dim"][0] * 1.1,params["dim"][1] * 1.1, params["dim"][2] * 1.1)
    pixel_array = skimage.transform.resize(pixel_array, output_shape, order=2, anti_aliasing=True)

    # t_2 = perf_counter()
    # print ("time to resize the image: ", t_2-t_1)
    # cropping
    # t_1 = perf_counter()

    depth_crop = (np.shape(pixel_array)[0] - params["dim"][0]) / 2
    lenght_crop = (np.shape(pixel_array)[1] - params["dim"][1]) / 2
    width_crop = (np.shape(pixel_array)[2] - params["dim"][2]) / 2

    pixel_array = pixel_array[int(depth_crop) :int(np.shape(pixel_array)[0] - depth_crop), \
                  int(lenght_crop) :int(np.shape(pixel_array)[1] - lenght_crop), \
                  int(width_crop) :int(np.shape(pixel_array)[2] - width_crop)]
    pixel_array = (pixel_array - np.mean(pixel_array)) / np.std(pixel_array)
    pixel_array = np.expand_dims(pixel_array, 3)
    #pixel_array =np.repeat(pixel_array,3,axis=-1)
    pixel_array = np.expand_dims(pixel_array, 0)
    print(np.shape(pixel_array))
    return pixel_array

image_path = "./storage/processed_data/axt2flair/_irw_BRAIN_DATA_irw_dcm_storage_2019_06_11_dcms_4067582579_1161711901_11.npz"
imagedata = np.load(image_path, allow_pickle=True)
pixel_array = resize(imagedata['pixel_array'])

print("Pixel mean: ", np.mean(pixel_array))
print("Pixer max/min:", np.max(pixel_array),np.min(pixel_array))

#predict
pred = model.predict(pixel_array)
print(np.shape(pred))
print("Prediction ", int(np.argmax(pred, axis=1)))


penultimate_layer = utils.find_layer_idx(model, last_conv_layer(model))