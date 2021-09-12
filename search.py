import os
import glob
import pathlib
from shutil import copyfile
from shutil import rmtree
import subprocess
import csv
from datetime import datetime
from pytz import timezone
import pytz

import tensorflow as tf
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa
import numpy as np

from configuration import DATA_FOLDER, OUTPUT_FOLDER


def get_model(model_path):
    model_keras = tf.keras.models.load_model(model_path)
    # saver.restore(sess, "/tmp/model.ckpt")
    return model_keras


def stable_softmax(X):
    exps = np.exp(X - np.max(X, 1).reshape((X.shape[0], 1)))
    return exps / np.sum(exps, 1).reshape((X.shape[0], 1))


def filename_to_localdatetime(filename):
    """
    Extracts datetime of recording in Costa Rica time from hexadecimal file name.
    Example call: filename_to_localdatetime('5A3AD5B6')
    """
    time_stamp = int(filename, 16)
    naive_utc_dt = datetime.fromtimestamp(time_stamp)
    aware_utc_dt = naive_utc_dt.replace(tzinfo=pytz.UTC)
    cst = timezone('America/Costa_Rica')
    cst_dt = aware_utc_dt.astimezone(cst)
    return cst_dt


def search_for_monkeys(wav_folder, output_folder, model_keras, threshold_confidence):
    # List all file names in folder
    file_name_list = glob.glob(wav_folder + '/*.WAV')
    file_name_list = [os.path.splitext(x)[0] for x in file_name_list]
    file_name_list = [os.path.basename(x) for x in file_name_list]

    for file_name in file_name_list:
        search_file_for_monkeys(file_name,
                                output_folder,
                                threshold_confidence=threshold_confidence,
                                wav_folder=folder,
                                model_keras=model_keras,
                                summary_file=True)


def search_file_for_monkeys(file_name,
                            output_folder,
                            threshold_confidence,
                            wav_folder,
                            model_keras,
                            tidy=True,
                            summary_file=False):
    """
    Splits 60-second file into 3-second clips. Runs each through
    detector. If activation surpasses confidence threshold, clip
    is separated.
    If hard-negative mining functionality selected, function
    takes combination of labelled praat file and 60-second wave file,
    runs detector on 3-second clips, and seperates any clips that
    the detector incorrectly identifies as being positives.
    These clips are then able to be fed in as negative examples, to
    improve the discriminatory capability of the network

    Example call: search_file_for_monkeys('5A3AD7A6', 60, '/home/dgabutler/Work/CMEEProject/Data/whinnies/shady-lane/')
    """
    audio_folder = wav_folder
    # isolate folder name from path:
    p = pathlib.Path(wav_folder)
    isolated_folder_name = p.parts[2:][-1]
    wav = audio_folder + "/" + file_name + '.WAV'
    print(wav)
    try:
        wavfile = AudioSegment.from_wav(wav)
    except OSError:
        print("\nerror: audio file", os.path.basename(wav), "at path", os.path.dirname(wav),
              "cannot be loaded - probably improperly recorded")
        return
    clip_length_ms = 3000
    clips = make_chunks(wavfile, clip_length_ms)

    # print("\n-- processing file " + file_name +'.WAV')

    clip_dir = wav_folder + '/clips-temp/'
    # delete temporary clips directory if interuption to previous
    # function call failed to remove it
    if os.path.exists(clip_dir) and os.path.isdir(clip_dir):
        rmtree(clip_dir)
    # create temporary clips directory
    os.makedirs(clip_dir)

    # Export all inviduals clips as wav files
    # print 'clipping 60 second audio file into 3 second snippets to test...\n'
    for clipping_idx, clip in enumerate(clips):
        clip_name = "clip{0:02}.wav".format(clipping_idx + 1)
        clip.export(clip_dir + clip_name, format="wav")

    D_test = []

    clipped_wavs = glob.glob(clip_dir + 'clip*')
    clipped_wavs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for clip in clipped_wavs:
        y, sr = librosa.load(clip, sr=48000, duration=3.00)
        # ps = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, win_length=1024, window='hamming')
        # ps = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)

        waveform = librosa.core.resample(y, orig_sr=48000, target_sr=16000)

        spectrogram = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=10 * 16)) ** 1.0
        logmel_spectrogram = librosa.power_to_db(
            librosa.feature.melspectrogram(y=waveform, sr=16000, S=spectrogram))
        logmel_spectrogram = logmel_spectrogram.transpose()[:-1, :]

        logmel_spectrogram_mean = np.mean(logmel_spectrogram, axis=0)
        logmel_spectrogram_std = np.std(logmel_spectrogram, axis=0)

        logmel_spectrogram_std[logmel_spectrogram_std == 0.0] = 1.0

        logmel_spectrogram = (logmel_spectrogram - logmel_spectrogram_mean) / logmel_spectrogram_std

        ps = logmel_spectrogram

        if ps.shape != (300, 128): continue
        D_test.append(ps)

    # D_test = wavtools.denoise_dataset(D_test)

    call_count = 0
    # reshape to be correct dimension for CNN input
    # NB. dimensions are: num.samples, num.melbins, num.timeslices, num.featmaps
    # print "...checking clips for monkeys..."
    for idx, clip in enumerate(D_test):
        D_test[idx] = clip.reshape(1, 300, 128)
        # D_test[idx] = np.concatenate([clip.reshape(1, 300, 128, 1),
        #                               clip.reshape(1, 300, 128, 1),
        #                               clip.reshape(1, 300, 128, 1)], axis=3)
        predicted = model_keras.predict(D_test[idx])

        # print()
        # print(predicted)
        predicted = stable_softmax(predicted)
        # print(predicted)

        # if "5C52E206" in file_name:
        #     print(predicted.shape)

        # if NEGATIVE:
        if predicted[0][1] <= (
                threshold_confidence / 100.0):  ########## THIS IS SECTION THAT CHANGED BETWEEN 1 node/2 node:
            continue  # WAS: if predicted[0][1] <= (threshold_confidence/100.0)
            # furthermore 3 changes (predicted[0][1] -> ..cted[0][0]) below
        else:
            # if POSITIVE
            call_count += 1
            lower_clip_bound = (3 * (idx + 1)) - 3
            upper_clip_bound = 3 * (idx + 1)
            # i.e. clip 3 would be 6-9 seconds into original 60-sec file
            approx_position = str(lower_clip_bound) + '-' + str(upper_clip_bound)

            # suspected positives moved to folder in Results, files renamed 'filename_numcallinfile_confidence.WAV'
            # results_dir = '/media/dgabutler/My Passport/Audio/detected-positives/'+isolated_folder_name+'-results'
            results_dir = output_folder + '/detected-positives/' + isolated_folder_name + '-results'

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            copyfile(clipped_wavs[idx],
                     results_dir + '/' + file_name + '_' + str(call_count) + '_' + approx_position + '_' + str(
                         int(round(predicted[0][1] * 100))) + '.WAV')

            # making summary file
            if summary_file:
                summary_file_name = output_folder + "/" + isolated_folder_name + '-results-summary.csv'
                # obtain datetime from file name if possible
                try:
                    datetime_of_recording = filename_to_localdatetime(file_name)
                    date_of_recording = datetime_of_recording.strftime("%d/%m/%Y")
                    time_of_recording = datetime_of_recording.strftime("%X")
                # if not possible due to unusual file name,
                # assign 'na' value to date time
                except ValueError:
                    date_of_recording = 'NA'
                    time_of_recording = 'NA'

                    # values to be entered in row of summary file:
                column_headings = ['file name', 'approx. position in recording (secs)', 'time of recording',
                                   'date of recording', 'confidence']
                csv_row = [file_name, approx_position, time_of_recording, date_of_recording,
                           str(int(round(predicted[0][1] * 100))) + '%']

                # make summary file if it doesn't already exist
                summary_file_path = pathlib.Path(summary_file_name)
                if not summary_file_path.is_file():
                    with open(summary_file_name, 'w') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter=',')
                        filewriter.writerow(column_headings)
                        filewriter.writerow(csv_row)

                # if summary file exists, *append* row to it
                else:
                    with open(summary_file_name, 'a') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter=',')
                        filewriter.writerow(csv_row)

    # delete all created clips and temporary clip folder
    if tidy:
        rmtree(clip_dir)
        # empty recycling bin to prevent build-up of trashed clips
        subprocess.call(['rm -rf /home/dgabutler/.local/share/Trash/*'], shell=True)


if __name__ == '__main__':
    MODEL_PATH = OUTPUT_FOLDER + "/SEResNet28-avg/" + "whinny_single" + "_model"
    # MODEL_PATH = "/path/to/where/model/is"

    FOLDER_LIST = [DATA_FOLDER + "/" + "test-monkey-corcovado3",
                   DATA_FOLDER + "/" + "test-monkey-will"]

    MODEL_KERAS = get_model(MODEL_PATH)

    THRESHOLD_CONFIDENCE = 60  # PER CENT

    for folder in FOLDER_LIST:
        search_for_monkeys(wav_folder=folder,
                           output_folder=OUTPUT_FOLDER,
                           model_keras=MODEL_KERAS,
                           threshold_confidence=THRESHOLD_CONFIDENCE)
