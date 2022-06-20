import tensorflow as tf

from model.audiorepresentation import AudioRepresentation
from model import mp3net
import utils.audio_utils as audio_utils

import os
import glob
import argparse

def audio2tfrecord(audio_file_paths: str, output_filename: str, model: mp3net.MP3netFactory, verbose: bool):
  """Pre-process audio files to tensors and write to output_filename

  :param audio_file_paths:         list of file paths of the input files
  :param output_filename:          output file name
  :param model:                    representation model to convert audio signal to audio representation.
  :return:                         nothing
  """
  audio_representation = AudioRepresentation(model.sample_rate, model.freq_n[-1], compute_dtype=tf.float32)

  # Create dataset of filepaths
  dataset = tf.data.Dataset.from_tensor_slices(audio_file_paths)

  # STEP 1. filepath to read in raw audio data
  def _decode_audio_shaped(filepath):
    _decode_audio_closure = lambda _filepath: audio_utils.load_audio(_filepath.numpy().decode('utf-8'),
                                                                     sample_rate=model.sample_rate,
                                                                     mono=(model.channels_n[-1]==1), verbose=verbose)
    audio = tf.py_function(_decode_audio_closure, [filepath], tf.float32)
    return audio

  dataset = dataset.map(_decode_audio_shaped, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # STEP 2. convert audio fragments to spectrograms
  @tf.function
  def wav_to_spectrogram(wav_data):
    # [samples_n, channels_n]
    if verbose:
        print('-'*20)
        print(wav_data)
        print(tf.shape(wav_data))
        print(model.channels_n)
        print('-'*20)

    tf.assert_equal(tf.shape(wav_data)[1], model.channels_n[-1],
                    f"Audio data has {tf.shape(wav_data)[1]} channels, but needs to have {model.channels_n[-1]} for the model")
    wav_data.set_shape([None, model.channels_n[-1]])

    # truncate audio (only a tiny bit)
    samples_n = tf.shape(wav_data)[0]
    last_block = tf.truncatediv(samples_n, model.freq_n[-1])
    wav_data = wav_data[:last_block*model.freq_n[-1], :]

    wav_data = tf.expand_dims(wav_data, axis=0)  # add batch dimension (t_to_repr expects it)
    mdct_norm = audio_representation.t_to_repr(wav_data)  # -1..1
    # [batches_n=1, blocks_n, freqs_n, channels_n]
    return mdct_norm

  dataset = dataset.map(wav_to_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # STEP 3. add masking threshold
  def add_masking_intensity(mdct_norm):
    mdct_norm.set_shape([1, None, model.freq_n[-1], model.channels_n[-1]])

    # compute masking threshold amplitude --> determines standard deviation of noise to be added
    masking_threshold = audio_representation.psychoacoustic_masking_ampl(mdct_norm)

    # remove batch_n dimension again
    masking_threshold = masking_threshold[0, :, :, :]
    mdct_norm = mdct_norm[0, :, :, :]

    return tf.stack([mdct_norm, masking_threshold], axis=-1)
  dataset = dataset.map(add_masking_intensity, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # STEP4. convert to string & write to file
  dataset = dataset.map(tf.io.serialize_tensor)
  writer = tf.data.experimental.TFRecordWriter(output_filename, compression_type="GZIP")
  writer.write(dataset)


def main(data_dir: str, data_destination: str, file_extension: str, batch_size: int, verbose: bool):
    model = mp3net.MP3netFactory()

    file_pattern = os.path.join(data_dir, f"*.{file_extension}")
    audio_files_paths = glob.glob(file_pattern)

    input_files_batched = [audio_files_paths[i:i + batch_size] for i in range(0, len(audio_files_paths), batch_size)]

    for batch_num, batch in enumerate(input_files_batched):
        if verbose:
            print(f"Batch {batch_num}")
        tf_output_filename = os.path.join(data_destination, f"yt-{batch_num:04d}" + f"_sr{model.sample_rate}" 
                                                        + f"_Nx{model.freq_n}x{model.channels_n}.tfrecord")
        if glob.glob(tf_output_filename):
            if verbose:
                print(f"Output file {tf_output_filename} already exists. Skipping...")
        else:
            audio2tfrecord(batch, tf_output_filename, model, verbose)
            if verbose:
                print(f"File {tf_output_filename} created.")
    if verbose:
        print("Data preprocessing complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Audio files directory to be transformed to TfRecords.", required=True)
    parser.add_argument("--data_dest", help="Destination directory where the TfRecords will be saved.", required=True)
    parser.add_argument("--file_extension", help="", default="wav")
    parser.add_argument("--batch_size", help="Batch size for grouping audio samples in TfRecords.", default=10, type=int)
    parser.add_argument("--verbose", help="Flag to turn console display on (is off by default).", default=False, action='store_true')
    args = parser.parse_args()
    main(args.data_dir, args.data_dest, args.file_extension, args.batch_size, args.verbose)