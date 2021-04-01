

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
#provided model and trained model with identical architecture
#provide 3D pixel_array as numpy array in the same shape as the input size of the 3D model
#pixel_array dimensions: (depth, length, width) or (1,depth, length, width,1); depth = dimension of interactive slider


import xai_vis.interact as interact

interact.vis(model,model_path,pixel_array)


```

## Examples

<img src="https://user-images.githubusercontent.com/51263484/112940011-cbe05f80-912c-11eb-97bd-7e776e645b65.png" width="500" height="360"> 
<img src="https://user-images.githubusercontent.com/51263484/112939970-b4a17200-912c-11eb-9c5b-ac51e0dfef12.png" width="500" height="360"> 















