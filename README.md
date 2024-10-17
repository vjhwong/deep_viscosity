# Deep Learning for Viscosity Estimations
This repository contains the code for the Deep Viscosity Project. The project aims at creating a 3D convolutional neural network that can predict viscosities of fluids based on video data of teh fluid in question. 

## Setting up a virtual environment
To create a Conda environment using the `environment.yml` file in this repository, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory where the `environment.yml` file is located.
3. Run the following command to create the Conda environment:

    ```
    conda env create -f environment.yml
    ```

4. Conda will now download and install all the necessary packages specified in the `environment.yml` file.
5. Once the environment creation process is complete, activate the environment using the following command:

    ```
    conda activate deep_viscosity
    ```

You have now successfully created and activated the Conda environment using the `environment.yml` file.


## Preprocessing the data
### The data
All raw data should be avi files stored in the `data/raw` folder. These videos need to be named according to the following convention: 
*(percentage of glycerol)_(test number).avi*

The allowed values for the percentages of glycerol can be found in the file `deep_viscosity/preprocessing/rename_data_files.py`.
### Preprocessing
To preprocess the data one should run the file `deep_viscosity/preprocessing/preprocessing.py`. Arguments must be passed to this script which should be done via the command line. The required arguments can be found in the file `deep_viscosity/preprocessing/utils/preprocess_args.py`. The exact command used for the preprocessing can be found in `deep_viscosity/arguments/preprocessing.txt`






