import argparse
from gtzan import AppManager

import glob
import numpy as np
import tqdm

# Constants
genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}

def compute_rockness_scores(folder, args):
    
    songs = glob.glob(folder + '/*.wav')
    app = AppManager(args, genres)
    rock_scores = []

    for i in tqdm.tqdm(range(len(songs))):
        app.args.song = f'{folder}/generated_clip_{i}.wav'
        preds = app.run()
        preds = np.mean(preds, axis=0)

        rock_score = preds[-1]
        rock_scores.append(rock_score)

        print(f"Rockness: {rock_score}")

    np.save(f'{folder}/rockness_scores', rock_scores)
    print((-np.asarray(rock_scores)).argsort()[:10])
    print((-np.sort(-np.asarray(rock_scores)))[:10])

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Music Genre Recognition on GTZAN')

    args = parser.parse_args()

    args.type = 'dl'
    args.model = '../models/custom_cnn_2d.h5'

    # Call the main function
    compute_rockness_scores('./data/acoustic', args)
    compute_rockness_scores('./data/all_instruments', args)
