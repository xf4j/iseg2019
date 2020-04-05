## Synopsis

Submission for MICCAI Grand Challenge on 6-month infant brain MRI Segmentation (http://iseg2017.web.unc.edu/). 3D U-Net and Dense-Net are implemented and tested. The goal is to segment gray matter (GM), white matter (WM) and cerebrospinal fluid (CSF) from T1- and T2-weighted images.</br><br/>
The challenge provided data with preprocessing described as follows:</br><br/>
For image preprocessing, T2 images were linearly aligned onto their corresponding T1 images. All images were resampled into an isotropic 1 × 1 × 1 mm3 resolution. Next, standard image preprocessing steps were performed before manual segmentation, including skull stripping, intensity inhomogeneity correction, and removal of the cerebellum and brain stem by using in-house tools. The preprocessing was conducted to maximally eliminate the influences of different image registration and bias correction algorithms on infant brain segmentation.</br><br/>
With raw data, skull stripping and intensity inhomogeneity correction can be performed using ants and nipype interface (https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.ants/segmentation.html). The removal of cerebellum and brain stem may not be necessary.

## Code Strcture

main.py: portal for invoking the training/cross-validation/testing process. For example, run `python main.py --phase=0 --model=UNet3D --config=models/UNet3D.ini --train_dir=../Training --test_dir=../Testing` to set the corresponding flags in the program. To use the default values defined in this script, the corresponding argument can be skipped.</br><br/>

models: stores all models and their corresponding config files.</br><br/>

models/base_model.py: base class for all models. Implement common functionalities such as save and load, mean and std estimation, data input/output and training and testing including augmentation. Note that if the data format is changed, the read_training_inputs and read_testing_inputs needs to be changed. Train and test are the main functions being called by the main.py script.</br><br/>

models/UNet3D.py: defines the model structure of UNet3D.</br><br/>

models/UNet3D.ini: defines configurable model parameters of UNet3D. This file is passed as a parameter to main.py and its content are read by UNet3D.py to build the corresponding model.</br><br/>

models/evaluations.py: helper functions for evaluations. To be improved for the 3D distance metrics since it takes too long to run now, possibly with reducing the number of boundary points.</br><br/>

After training, two folders will be created: logs and checkpoints, where the log files and model files are saved, respectively. The names and locations of the two folders can be changed by changing the variables defined in main.py, however this is usually unnecessary.</br><br/>

It is recommended to have a script called pre_processing.py to perform all image preprocessing. Contact the author for help and/or examples.

## Installation

The model is implemented and tested using `python 2.7` and `Tensorflow 1.14.0`, but `python 3` and newer versions of `Tensorflow` should also work.</br><br/>
Other required libraries include: `numpy`, `h5py`, `skimage`, `transforms3d`, `SimpleITK`. Simply run `pip install xxx` to install the missing library. Use `pip install --user xxx` to install under home folder. Currently there are no issues identified with different versions of the third-party libraries. Contact the author if any issues are found.

## Contributors

Xue Feng, Department of Biomedical Engineering, University of Virginia
xf4j@virginia.edu