## Acknowlegment
We expect everything to work on an isolated python environment created 
as per the instructions below, but in case you face any issues running
the code please feel free to contact us by email or on MS-Teams ().

We have tested our code in an environment with the following specifications:
- Machine:
    - CPU: `11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz`
        - `x86_64 ` 
    - RAM: 16 GB
- OS: `Ubuntu 20.04.4 LTS`
- Python Version: `3.6.13`
## Importing the conda environment

In the command line, create a conda environment with name `ENVNAME` running:

    conda env create -f environment.yml -n ENVNAME
    conda activate ENVNAME

[TODO:] - Just give the requirements


# Overview of Code Structure


- The modules containing models are prefixed with `model_` except for 
`model_factory.py`, which contains classes to resolve and load models by name.
We experimented with different models and all of them can be found in these
files (however we have mentioned only the relevant ones in the report.)

- `data_loader.py` contains dataloaders for the two datasets and some extensions
thereof, to provide support for loading , splitting the subsampling the data.

- `trainingutil.py` : This contains the trainer (`#TODO`) classes to abstract
training loop and experiment pipeline classes 
(_e.g._ `ExperimentPipelineForClassifier`) to abstract away logging, data loading,
optimizer & cost_function iniatilazion _etc._ and provide and interface to 
run the training based on a single configuration file (refer below).

# How to Run?

## For PyTorch based Models:

- N.B. Most of our models are implemented in PyTorch and the steps below
apply to all such models

> Steps for training:
- Make sure you are inside the `src` directory
- To start training execute:
  - ```
    python trainingutil.py --config <path-to-run-config-file>
    ```
  - _e.g._
    ```
    python trainingutil.py --config experiment_configs/experiment_0_a_vanilla_cnn_mitbih.yaml
    ```
  - The `experiment_configs` folder contains many configs, that we have used
  for running our experiments. You can choose any of those or create your own

The steps above will do the following:
- It will start training 
- create `runs` folder if not already present
- create a timestamped folder with `tag` value provided in the config as suffix
_e.g._ : `2022-03-29_014835__exp_0_b_VanillaCnnPTB`
  - this folder will be used to output the best model 
  - in this folder `logs` subfolder will be created in which tensorboard logs
    and AUROC and AUPRC curves will be saved.
- the best model will be saved  if the validation F1 has increased when
  compared to the last best F1

> Steps for evaluating saved model:

