# Multi-Attentive Detection of the Spider Monkey Whinny in the (Actual) Wild.

Project repository for the Interspeech 2021 paper by Georgios Rizos, Jenna Lawson, Zhuoda Han, Duncan Butler, James Rosindell, Krystian Mikolajczyk, Cristina Banks-Leite, Bjoern W. Schuller

https://www.isca-speech.org/archive/interspeech_2021/rizos21_interspeech.html

## General description

The code classified 3 second audio clips as containing a spider monkey whinny or not (binary classification).

The preprocessing of the raw acoustic data constitutes normalisation, and storage as TF RECORDS files.

The training/evaluation loop used the stored TF RECORDS files.

## Dataset

Please contact Dr. Jenna L. Lawson for access to the dataset.

## How to use the repo

Follow these steps for replicating the experiments of the Interspeech-21 paper:
- Install all dependencies, summarised in requirements.txt
- Open configuration.py and edit the PROJECT_FOLDER and DATA_FOLDER variables. The first should point to this code, and the second to the Spider Monkey whinny dataset.
- Execute clip_and_store.py -- this segments the original audio clips into 3 second clips, extracts features, performs normalisation, and stores the clips as TF RECORDS files. This may require an hour of processing time.
- Run train.py to train the best performing model in the paper on the data.

The training configuration is summarised in corresponding YAML files in the experiment_configurations folder.

If you would like to run the other models from the paper, you can edit train.py such that it points to a different YAML file.

## How to add new training data

Edit and then run the add_training_files.py script.

This expects that you already have stored 3 second audio clip files in a specific folder. It will read all clips from a folder, extract features, and store as TF RECORDS files.

You need to edit the NEW_INPUT_FOLDER, ARE_POSITIVES, and SAMPLING_RATE variables in the script.
