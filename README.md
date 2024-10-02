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
All raw data should be avi files stored in the `data/raw` folder. 



