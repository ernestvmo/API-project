from pydub import AudioSegment
import os
import argparse


def slice_audio(audio_file: str, slice_len: float = 1) -> list:
    """The provided file is sliced into the number of chunks specified, then returns a list of all the smaller audio samples.

    Args:
        audio_file (str): Path to the audio file to be sliced.
        slice_len (int, optional): The size of a chunk of the sampled audio, in seconds. Defaults to 1.

    Returns:
        list: A list of all the audio sample chunks.
    """
    audio = AudioSegment.from_wav(audio_file)
    slices = []

    for i in range(0, len(audio), int(slice_len * 1000)):
        if not i == round(len(audio) / 1000) * 1000:
            slices.append(audio[i:i + int(slice_len * 1000)])

    return slices

def export_slices(slices: list, initial_sample_name: str, saving_location: str):
    """Exports the sliced audio sample to the specified destination directory.

    Args:
        slices (list): The chunks of the sliced audio sample.
        initial_sample_name (str): Filename of the initial audio sample (used for saving).
        saving_location (str): The directory for the export destination.
    """
    for i in range(len(slices)):
        slices[i].export(saving_location + initial_sample_name[:-4] + f"_{(i + 1)}.wav", format="wav")

def main(audio_files_path: str = None, export_path: str = "../data/data_sliced/two", slice_length: float = 1):
    if audio_files_path is None:
        assert ValueError("The path to the data should not be None.")

    audio_files = [elem.strip() for elem in os.listdir(audio_files_path)]

    for audio_file in audio_files:
        slices = slice_audio(audio_file=audio_files_path + audio_file, slice_len=slice_length)
        export_slices(slices, audio_file, export_path)
        print("Audio correctly sliced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice_len", "-s", help="The length (in second) of each chunk of the sliced audio sample.", type=float)
    parser.add_argument("--data_loc", "-dl", help="The directory containing the audio samples.")
    parser.add_argument("--export_loc", "-el", help="The directory where the sliced audio samples will be stored.")
    args = parser.parse_args()
    main(args.data_loc, args.export_loc, args.slice_len)