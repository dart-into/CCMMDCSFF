# Chrysanthemum Classification Method via Multi-Stream Deep Color Space Feature Fusion
## File Introduction
### main.py
main.py is the executable 
### load_front_gray_HSL_2_shuffle.csv
The file load_front_gray_HSL_2_shuffle.csv contains data. Columns 1 to 20 represent the data of chrysanthemum images processed through the H color channel. Columns 21 to 40 represent the data of chrysanthemum images processed through the S color channel. Columns 41 to 60 represent the data of chrysanthemum images processed through the L color channel. The last two columns respectively contain the absolute addresses of the images and their labels.
### model.pth
model.pth is a filename used to store PyTorch model files.This file contains the model's parameters, structure, and other necessary information for loading the model later to make predictions or continue training.
### model_state.pth
model_state.pth is a filename used to store the parameter state dictionary of a PyTorch model. This file contains only the parameters of the model, without the model's structure. As a result, it is relatively lightweight. By saving the model's state dictionary, we can recreate the model later and load the previously saved parameters without needing to redefine the model's structure.
## package
### The required libraries are as follows (some of which may not be useful, for reference only):
### Package                 Version
### ----------------------- ------------
### absl-py                 0.15.0
### astunparse              1.6.3
### bleach                  1.5.0
### cached-property         1.5.2
### cachetools              4.2.4
### certifi                 2021.5.30
### charset-normalizer      2.0.12
### clang                   5.0
### click                   8.0.4
### colorama                0.4.5
### cycler                  0.11.0
### dataclasses             0.8
### decorator               4.4.2
### enum34                  1.1.10
### filelock                3.4.1
### flatbuffers             1.12
### gast                    0.4.0
### google-auth             1.35.0
### google-auth-oauthlib    0.4.6
### google-pasta            0.2.0
### grpcio                  1.48.2
### h5py                    3.1.0
### html5lib                0.9999999
### huggingface-hub         0.4.0
### idna                    3.4
### imageio                 2.15.0
### importlib-metadata      4.8.3
### importlib-resources     5.4.0
### joblib                  1.1.1
### jsonpatch               1.32
### jsonpointer             2.3
### keras                   2.6.0
### Keras-Preprocessing     1.1.2
### kiwisolver              1.3.1
### Markdown                3.3.7
### matplotlib              3.3.4
### mkl-fft                 1.3.0
### mkl-random              1.1.1
### mkl-service             2.3.0
### networkx                2.5.1
### numpy                   1.19.5
### oauthlib                3.2.2
### olefile                 0.46
### opencv-python           4.6.0.66
### opt-einsum              3.3.0
### packaging               21.3
### pandas                  1.1.5
### Pillow                  8.4.0
### pip                     21.2.2
### protobuf                3.19.6
### pyasn1                  0.5.0
### pyasn1-modules          0.3.0
### pyparsing               3.1.1
### python-dateutil         2.8.2
### pytz                    2023.3.post1
### PyWavelets              1.1.1
### PyYAML                  6.0.1
### regex                   2023.8.8
### requests                2.27.1
### requests-oauthlib       1.3.1
### rsa                     4.9
### sacremoses              0.0.53
### scikit-image            0.17.2
### scikit-learn            0.24.2
### scipy                   1.5.4
### setuptools              58.0.4
### six                     1.16.0
### tensorboard             2.6.0
### tensorboard-data-server 0.6.1
### tensorboard-plugin-wit  1.8.1
### tensorflow              1.4.0
### tensorflow-estimator    2.6.0
### tensorflow-tensorboard  0.4.0
### termcolor               1.1.0
### threadpoolctl           3.1.0
### tifffile                2020.9.3
### timm                    0.8.11.dev0
### tokenizers              0.10.3
### torch                   1.10.2
### torchaudio              0.10.2
### torchvision             0.11.3
### tornado                 6.1
### tqdm                    4.64.1
### transformers            4.16.2
### typing-extensions       3.7.4.3
### urllib3                 1.26.16
### visdom                  0.2.4
### websocket-client        1.3.1
### Werkzeug                2.0.3
### wheel                   0.37.1
### wincertstore            0.2
### wrapt                   1.12.1
### zipp                    3.6.0
