# GEHENNUM
[![Version](https://img.shields.io/badge/version-1.0-blue.svg)](https://huggingface.co/Wintermute/Wintermute)
## What am I?
I am an A.I, created with the DCGAN architecture, and trained with occult and obscure art imagery.
## What is my purpose ?
To generate nightmarish and dark images.
## What is this repository?
This repository will contain the source code for the model.
## Usage
```python
>>> generator = generator()
>>> discriminator = discriminator()
>>> generator.load_state_dict(torch.load("gehennum_generator.ckpt"))
>>> discriminator.load_state_dict(torch.load("discriminator_generator.ckpt"))
```
The jupyter notebook used for training is also available for a more in depth understanding of the model.

Alternatively, you can run the [model.py](src/model.py) file for direct image generation:
```python
python model.py
```
## Info
The model was trained using a DCGAN architecture for 64x64 image generation. I used the [best artworks of all time dataset](https://www.kaggle.com/ikarus777/best-artworks-of-all-time), together with a 4000 image dataset scraped from pinterest, consisting of occult and dark imagery. These images will become available soon.

The model will soon be trained on a 256x256 architecture.
