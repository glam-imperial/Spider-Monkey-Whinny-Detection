import os
import collections

import librosa
import numpy as np

import wavtools, configuration
from common import data_sample, normalise, tfrecord_creator

########################################################################################################################
# Just edit these to point to the folder of new 3sec sound files.
########################################################################################################################
NEW_INPUT_FOLDER = "/path/to/new/input/files"

ARE_POSITIVES = True  # False if negatives.

SAMPLING_RATE = 48000

NAME = "new_data_0"  # Give a name to the group of data. Will appear in the TF-RECORDS file names. Not important re ML.

########################################################################################################################
# The rest reads the files and turns then into tf_records files.
########################################################################################################################

file_names = os.listdir(NEW_INPUT_FOLDER)
file_paths = [NEW_INPUT_FOLDER + "/" + name for name in file_names]


def sound_data_generator(file_paths, name, are_positives, sr):
    for idx, file_path in enumerate(file_paths):
        waveform, sr = librosa.core.load(file_path,
                                         sr=sr,
                                         duration=3.00)

        x_dict, custom_stats = wavtools.get_features_and_stats(waveform)

        if x_dict["logmel_spectrogram"].shape[0] != 300:
            print("Warning -- the sound file is not 3seconds long, or the sampling rate provided was incorrect.")
            print("Given sampling rate, the duration of clip is:", x_dict["logmel_spectrogram"].shape[0] / 100)
            print("Skipping data sample.")
            continue

        id_dict = collections.OrderedDict()
        id_dict["segment_id"] = idx
        id_dict["version_id"] = 0

        y_dict = dict()
        y_dict["whinny_single"] = np.zeros((2,), dtype=np.float32)
        if are_positives:
            y_dict["whinny_single"][1] = 1.0
        else:
            y_dict["whinny_single"][0] = 1.0

        # Continuous is unused.
        whinny_continuous_size = x_dict["waveform"].shape[0] * x_dict["waveform"].shape[1]
        whinny_continuous = np.zeros((whinny_continuous_size, 2), dtype=np.float32)
        y_dict["whinny_continuous"] = whinny_continuous

        support = np.ones((whinny_continuous_size, 1), dtype=np.float32)

        # Make DataSample.
        if are_positives:
            prefix = "pos_"
        else:
            prefix = "neg_"
        sample = data_sample.Sample(name=prefix + name,
                                    id_dict=id_dict,
                                    partition="train",
                                    x_dict=x_dict,
                                    y_dict=y_dict,
                                    support=support,
                                    is_time_continuous=False,
                                    custom_stats=custom_stats)

        yield sample


########################################################################################################################
# Normalisation, and tf record creation.
########################################################################################################################
generator = sound_data_generator(file_paths, NAME, ARE_POSITIVES, SAMPLING_RATE)

normaliser = normalise.Normaliser(sample_iterable=generator,
                                  normalisation_scope="sample")
normalised_sample_generator = normaliser.generate_normalised_samples()

tfrecord_creator = tfrecord_creator.TFRecordCreator(tf_records_folder=configuration.TFRECORDS_FOLDER,
                                                    sample_iterable=normalised_sample_generator,
                                                    are_test_labels_available=True,
                                                    is_continuous_time=False)
tfrecord_creator.create_tfrecords()
