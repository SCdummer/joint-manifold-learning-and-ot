# Joint Manifold Learning and Optimal Transport for Dynamic Imaging
This repository contains the code of the paper 'Joint Manifold Learning and Optimal Transport for Dynamic Imaging'. 

## Required packages:
The required packages are:
```
python=3.12.2
pytorch=2.5.1 (cuda 11.8)
pykeops=2.2.3
torchdiffeq=0.2.5
scikit-learn=1.6.1
scikit-image=0.25.0
matplotlib=3.10.1
tqdm=4.67.1
geomloss=0.2.6
einops=0.8.1
opencv-python=4.11.0.86
```
Optionally, one can create a conda environment that contains the required packages via the supplied `environment.yml` file:
```
conda env create -f environment.yml
```

## Training a model
For every experiment, you should create a specific directory where you store a `specs.json` file. The `specs.json` file should contain all the parameters for training. For examples, see the `experiments` folder. 


For a given experiment directory 'experiment_directory', for instance "Experiments/Cells/OT", you run the experiment by executing:

```
python train.py -e experiment_directory
```

## Creating datasets
The datasets should be stored in a subfolder of the `data` folder. 

**NOTE:** if we have a dataset called `example-dataset`, `data/.../example-dataset` should contain:
- folders `data/.../example-dataset/Track...`, which contain the tracks.
- a `split.json` file indicating which tracks belong to the training set and which to the test set.

Examples are given below.

### Creating/downloading the Gaussian dataset
One can download our used Gaussian dataset from .... This contains three folders: images (containing the images of the tracks), gifs (containing animations of the ground truth over time), and latents (the 2d variables used to generate each picture). For more information on the 'latent codes', we advise to look at the `src.data.create_gaussian_data.py` code. After downloading the dataset, make sure the `data/Gaussians/split.json` file is moved to `data/Gaussians/images/split.json`.

Alternatively, one can create their own Gaussian dataset by running the `src.data.create_gaussian_data.py` file via `python src/data/create_gaussian_data.py` and modifying the options in the file. Currently, this saves the synthetic data in the `data/Gaussians` folder. However, one can specify a new folder in the `create_gaussian_data.py` function. 

**NOTE:** when running `create_gaussian_data.py`, sometimes an error appears. If this happens, rerun the file until no error occurs.

### Creating cell image dataset
Load the zip file from the 
[Cell Tracking Dataset](http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip). 
Put this data in the data folder under `Fluo-N2DL-HeLa`.
Then run the following: 
````
python src/data/process_cell_tracking_data.py
````
This processes and saves the dataset in the subfolder `01_processed` of `Fluo-N2DL-HeLa`. After this you can remove the initial raw data files from the `Fluo-N2DL-HeLa` folder. Finally, make sure the `data/Fluo-N2DL-HeLa/split.json` file is moved to `data/Fluo-N2DL-HeLa/01_processed/split.json`.

## Specs.json file options
- **DataSource**: where the data for training is saved. E.g. "data/Fluo-N2DL-HeLa/01_processed".
- **DataSourceTest**: where the data for testing is saved. E.g. "data/Fluo-N2DL-HeLa/01_processed". The data source put here can be different from *DataSource* as *DataSource* might contain noisy samples while *DataSourceTest* contains the non-noisy samples. 
- **EncoderSpecs**: (specification of encoder) 
  - **num_filters**: the number of convolutional layers
  - **img_dims**: input dimensions of the image (e.g. [1, 64, 64])
  - **use_vae**: whether to use a VAE (true) or a deterministic AE (false). *(In the experiments in the paper it is always fixed to false)*
- **DecoderSpecs**: (specification of decoder)
  - **num_filters**: the number of transposed convolutional layers.
  - **most_common_val**: the value between 0.0 and 1.0 around which we center the output values of the decoder. In case, 0.0 < most_common_val < 1.0, the decoder output *out* is of shape (..., 2). Then *out*[..., 0] determines the value in the interval [0.0, most_common_val] and *out*[..., 1] the value in [most_common_val, 1.0]. In case most_common_val==0, the decoder output is of shape (..., 1). *(In the experiments in the paper it is always fixed to 0.0)*
- **TimeWarperSpecs**: (specification of neural ode)
  - **hidden_dims**: the number of hidden layers
  - **ode_int_method**: the method of integration
  - **act**: the activation function used
  - **adjoint_method**: whether to use the adjoint method for backpropagation or whether to use backpropagation through ode solver operations.
- **EvalOn**: on what data we evaluate during training (can be "train" or "val").
- **TimeSubSampling**: e.g. if the value is 5, we take the images at $t_0$, $t_5$, $t_{10}$, ...
- **NumIntSteps**: the number of integration steps to go from e.g. $t_0$ to $t_1$ (or from $t_i$ to $t_{i+1}$). This is only relevant for explicit methods such as Euler's method.
- **Nabla_t**: the actual time between two timepoints. So $t_{i+1}-t_i = $ Nabla_t.
- **NumEpochsStatic**: the number of epochs in which we ONLY train the encoder and decoder.
- **NumEpochsDynamic**: the number of epochs in which we train the neural ode and, optionally, the encoder and decoder.
- **JointLearning**: whether to learn the encoder and decoder when learning the neural ode.
- **BatchSizeStatic**: the amount of samples to grab during static learning.
- **BatchSizeDynamic**: the amount of samples to grab during dynamical learning.
- **LatentDim**: the used latent dimension.
- **InitLR**: initial learning rate of the ADAM optimizer used.
- **SaveFreq**: after how many epochs to save AND evaluate the model (on the training data).
- **UseWandb**: whether to use weights and biases or not.
- **Continue**: when you have saved a previous model, whether to continue from the last saved checkpoint.
- **LambdaReconDynamic**: we have a static reconstruction and dynamic reconstruction loss. This value is the contant in front of the dynamic reconstruction loss.
- **NumRegPoints**: for the regularizers we regularize at points in between consecutive training times AFTER subsampling. E.g. if we have *TimeSubSampling* $=5$ and we consider $t_5$ and $t_{10}$, we grab equidistant points in the interval $[t_5, t_{10}]$ and *NumRegPoints* indicates the number of equidistant points to use. 
- **LambdaDynRegLat**: the constant for $\lVert \frac{d}{dt} z(t) \rVert _2 $ regularization
- **LambdaDynRegL2**: the constant for $\lVert \frac{d}{dt} D(z(t)) \rVert _2 $ regularization
- **LambdaDynRegOT**: the constant for OT regularization.

## Evaluating the trained model
To evaluate a trained model, you can execute the following command in the command line: 
```
python -m src.data.evaluate_model.py -e experiment_directory -m max_val
```
where max_val is used to normalize the images (and not the gifs):
- *max_val*="time_series": this means we normalize all the images with the maximum value across all images in the (reconstructed) time series.
- *max_val* is a float (e.g. 0.7): this means we scale all the images by 1/max_val and then plot the values in the interval [0.0, 1.0] (so the values >1 are assigned the value 1)

The code produces (on both the training and test set and for both dynamic and static reconstructions):
- Evaluation metrics for the reconstructions.
- Animations in .gif format that shows how the reconstructions (either static or dynamic) develops over time and how it compares to the ground truth solution.
- A plot of the full latent space (obtained using PCA if latent dimension is unequal to 2) and of the individual trajectories in this (PCA) space.
- Individual images for each time point in the reconstruction of the track. (these images are scaled with the *max_val* input)

## Some other functions

### `src.evaluation.barycentric_interpolation.py`
This code can be used to generate $l_2$ or $\mathcal{W}_2$ (Wasserstein) interpolations between two samples. It can be run via:
```
python -m src.evaluation.barycentric_interpolation -f1 path_to_img1 -f2 path_to_imgs2 -s save_dir -m max_val
```
where
- path_to_img1 denotes the path to the left end point of the interpolation.
- path_to_img2 denotes the path to the right end point of the interpolation.
- save_dir denotes the directory where the $\mathcal{W}_2$ and $l_2$ interpolations are saved. In particular, they are saved in save_dir/OT and save_dir/l2, respectively.
- max_val denotes the maximum value that is used when plotting the images. 

### `src.evaluation.images_to_matplotlib_pngs.py`
This code creates png images of the ground truth data via matplotlib. This might be useful when comparing the reconstruction to the ground truth. 
The parameters can be changed inside the function itself. 

## Some final remark
The code has only been tested on Linux. If you encounter problems on Windows, we suggest using Windows Subsystem for Linux (WSL2). 

## Citation
When using this code, please reference the following paper:
```
@article{zeune2020deep,
  title={Deep learning of circulating tumour cells},
  author={Zeune, Leonie L and Boink, Yoeri E and van Dalum, Guus and Nanou, Afroditi and de Wit, Sanne and Andree, Kiki C and Swennenhuis, Joost F and van Gils, Stephan A and Terstappen, Leon WMM and Brune, Christoph},
  journal={Nature Machine Intelligence},
  volume={2},
  number={2},
  pages={124--133},
  year={2020},
  publisher={Nature Publishing Group UK London}
}
```