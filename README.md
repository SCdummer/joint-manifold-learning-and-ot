# Spatiotemporal imaging and image prognosis via Manifold learning in Wasserstein space

How to run an experiment:
```
python train.py -e experiment_directory
```

## Required packages:
- pytorch
- torchdiffeq
- tqdm
- scipy
- scikit-learn
- pykeops
- numpy
- matplotlib
- geomloss
- wandb

I also attached an `environment.yml` file that can be used to create a conda environment.

## Creating cell image dataset
Load the zip file from the 
[Cell Tracking Dataset](http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip). 
Put this data in the data folder under `Fluo-N2DL-HeLa`.
Then run the following: 
````
python src/data/process_cell_tracking_data.py
````
This processes and saves the dataset in the subfolder `01_processed` of `Fluo-N2DL-HeLa`.

## Specs.json file options
- **DataSource**: where the data for training is saved. E.g. "data/Fluo-N2DL-HeLa/01_processed".
- **EncoderSpecs**: (specification of encoder) 
  - **num_filters**: the number of convolutional layers
  - **img_dims**: input dimensions of the image (e.g. [1, 64, 80])
  - **use_vae**: whether to use the vae variant or not.
- **DecoderSpecs**: (specification of decoder)
  - **num_filters**: the number of transposed convolutional layers.
- **TimeWarperSpecs**: (specification of neural ode)
  - **hidden_dims**: the number of hidden layers
  - **ode_int_method**: the method of integration
  - **act**: the activation function used
  - **adjoint_method**: whether to use the adjoint method for backpropagation or whether to use backpropagation through ode solver operations.
- **TimeSubSampling**: e.g. if the value is 5, we take the images at $t_0$, $t_5$, $t_10$, ...
- **NumIntSteps**: the number of integration steps to go from e.g. $t_0$ to $t_1$ (or from $t_i$ to $t_{i+1}$). This is only relevant for explicit methods such as Euler's method.
- **Nabla_t**: the actual time between two timepoints. So $t_{i+1}-t_i = $Nabla_t.
- **NumEpochsStatic**: the number of epochs in which we ONLY train the encoder and decoder.
- **NumEpochsDynamic**: the number of epochs in which we train the neural ode and, optionally, the encoder and decoder.
- **JointLearning**: whether to learn the encoder and decoder when learning the neural ode.
- **BatchSize**: the amount of samples to grab during static learning and dynamical learning.
- **LatentDim**: the used latent dimension.
- **InitLR**: initial learning rate of the ADAM optimizer used.
- **SaveFreq**: after how many epochs to save AND evaluate the model (on the training data).
- **UseWandb**: whether to use weights and biases or not (currently I never used this).
- **Continue**: when you have saved a previous model, whether to continue from the last saved checkpoint.
- **h**: NOT USED AT THE MOMENT.
- **NumRegPoints**: NOT USED AT THE MOMENT.
- **lambda_dyn_reg**: NOT USED AT THE MOMENT.