

# XAI visualisation

Visualisaton of interpretability methods of 3D MRI images in Jupyter Notebook for Tensorflow

Interpretablity methods derived from https://github.com/keisen/tf-keras-vis

## Requirements

```
Python 3.8
TensorFlow 2.2.0
Keras 2.4.3
tf-keras-vis 0.5.5
```

## Installation


```bash
pip install git+https://github.com/mariusberlin/xai_visualisation
```

## Usage

```python
#The calculation of the saliency maps can take up to 10 minutes depending on your GPU.
#Provide model, trained model and pixel array in a Jupyter Notebook

import xai_vis.utils
import xai_vis.methods
import xai_vis.interact as interact


#your model architecture
model = vgg16_model((30, 128, 128,1), 64, 2, 0.2, 2)

#path to trained model
path_name = "./storage/trained_models/t2_flair/2021-02-12 22:06:02_final_augm0.7_dim_(30, 128, 128)_lr_decay_plateau_npz_axt2flair_vgg16_final_adam_reg_0_dropout_0.4_lr_0001_val_loss.hdf5"

or
#trained model
#path_name = trained_model

xai_vis.interact.vis(model,model_path,pixel_array)


```

## Examples

<img src="https://user-images.githubusercontent.com/51263484/112940011-cbe05f80-912c-11eb-97bd-7e776e645b65.png" width="500" height="360"> 
<img src="https://user-images.githubusercontent.com/51263484/112939970-b4a17200-912c-11eb-9c5b-ac51e0dfef12.png" width="500" height="360"> 















