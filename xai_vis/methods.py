from tf_keras_vis.scorecam import ScoreCAM
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tensorflow.keras import backend as K
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam
import numpy as np

'''
Call the interpretablity methods
Interpretability methods are from: https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb
'''

'''
Vanilla Saliency
'''
def call_vsaliency(model,model_modifier,loss,pixel_array):
    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)

    # Generate saliency map
    saliency_map = saliency(loss, pixel_array)
    return normalize(saliency_map)

'''
Saliency Smooth
'''
def call_smooth(model,model_modifier,loss,pixel_array) :
    # Create Gradcam object
    # The `output` variable refer to the output of the model,
    # i.e., (samples, classes). either output[0][1] or output[0][0]: output[sample_idx][class_idx]
    def loss(output):
        return (output[0][1])  # sample 0 class 1
    
    saliency = Saliency(model,
                    model_modifier=model_modifier,
                    clone=False)

    # Generate saliency map
    saliency_map = saliency(loss, pixel_array, smooth_samples=10,  # The number of calculating gradients iterations.
                            smooth_noise=0.20)  # noise spread level.)

    capi = normalize(saliency_map)
    #print("Shape normalize Cam: ", np.shape(capi))
    return capi

'''
Smooth grad-cam 
'''
def call_grad(model,model_modifier,loss,pixel_array):
    gradcam = Gradcam(model,
                      model_modifier=model_modifier,
                      clone=False)

    # Generate heatmap with GradCAM
    cam = gradcam(loss,
                  pixel_array,
                  penultimate_layer=-1, # model.layers number
             )
    return normalize(cam)

'''
GradCam PlusPlus
'''
def call_gradplus(model,model_modifier,loss,pixel_array) :
    gradcam = GradcamPlusPlus(model,
                              model_modifier,
                              clone=False)

    # Generate heatmap with GradCAM++
    cam = gradcam(loss,
                  pixel_array,
                  penultimate_layer=-1,  # model.layers number
                  )
    return normalize(cam)

'''
Faster Score-Cam
'''
def call_faster_scorecam(model, model_modifier,loss,pixel_array):

    # Create ScoreCAM object
    scorecam = ScoreCAM(model, model_modifier, clone=False)

    # Generate heatmap with Faster-ScoreCAM
    cam = scorecam(loss,
                   pixel_array,
                   penultimate_layer=-1, # model.layers number
                   max_N=10
                   )
    return normalize(cam)
