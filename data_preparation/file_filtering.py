import os
import shutil
import argparse


def filter_and_copy_files(source: str, destination: str):
    """This method will go through all the files in the provided directory and filter out the files that do not have the tag 'pop_roc' in the filename. All those files are then copied to the provided destination directory.

    Args:
        source (str): Data directory of the audio samples to filter.
        destination (str): Directory where the filtered files should be copied to.

    Raises:
        ValueError: In the event not all files where copied to the destination directory, the function raises an Error.
    """
    audio_files = []

    for instru_dir in os.listdir(source):
        for audio_file in os.listdir(source + instru_dir):
            if "[pop_roc]" in audio_file:
                audio_files.append(instru_dir + '/' + audio_file)
                shutil.copyfile(src='/'.join([source, instru_dir, audio_file]), dst='/'.join([destination, audio_file]))
    if not len(os.listdir(destination)) == len(audio_files):
        raise ValueError("SOMETHING WENT WRONG")
    else:
        print(f"{len(audio_files)} elements were copied to {destination}")

def main(audio_files_path: str = None, export_path: str = "../data/irmas/"):
    if audio_files_path is None:
        assert ValueError("The path to the data should not be None.")
    filter_and_copy_files(audio_files_path, export_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc", "-dl", help="The directory containing the audio samples.")
    parser.add_argument("--export_loc", "-el", help="The directory where the sliced audio samples will be stored.")
    args = parser.parse_args()
    main(args.data_loc, args.export_loc)