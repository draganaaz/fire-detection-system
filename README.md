# Planet Balkan Hackaton
## Fire detection and prevention system 

We are building a tensorflow image classification python app that will differentiate satellite imagery containing smoke from those which do not

## Verisons
Python - 3.8.3

Tensorflow - 2.3.1

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorflow.

```bash
pip install tensorflow
```

## Images 
![Alt text](https://i.imgur.com/nvyGLgZ.jpg)

### First image output:
Prediction for the No_Smoke_1 without smoke: It most likely has No Smoke with a **99.6%** percent confidence.

### Second image output:
Prediction for the Smoke_5 with smoke: It most likely has Smoke with a **98.2%** percent confidence.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Attention
Due to lack of data for model training, our model will most likely classify few images containing smoke as image with no smoke, and vice-verse.
