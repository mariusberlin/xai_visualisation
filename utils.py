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

# define the output class: sample 0/class 1,
def loss(output ):
    return (output[0][1])  


def model_modifier(m ):
    m.layers[-1].activation = tf.keras.activations.linear
    return m


#funktion to return max slice or a slice view
def mode(mode, capi) :
    
    if mode == "max" :
        capi = np.squeeze(capi)
        capi_gap = np.apply_over_axes(np.mean, capi, [1, 2])
        capi_gap_max = np.argmax(capi_gap)
        return np.squeeze(capi[capi_gap_max, :, :])

    elif mode == "slice" :
        capi = np.squeeze(capi)
        return capi
    
    '''
    elif mode == "mean" :
        capi = np.squeeze(capi)
        capi = np.mean(capi, axis=0)
        # print("mean capi: ",(capi))
        return capi
    '''
