# API Project

This project is created for the seminar course Audio Processing and Indexing at Leiden University.<br>
In this project, we aim to investigate state of the art GAN networks to generate rock music audio samples.<br>

<!-- ### Run Code
```
# connect
ssh s3210359@ssh.liacs.nl
ssh duranium
# 
``` -->

### Datasets
To train the models, we used the following datasets:
  - [GTZAN Dataset-Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download): this dataset, available freely on Kaggle, contains 10000 files of 30 seconds audio samples, divided equally into 10 different music genres. Taking only the rock genre, we can use the 100 samples of 30 seconds to divide them into smaller more usable samples.
  - [IRMAS](https://www.upf.edu/web/mtg/irmas): Used for Instrument Recognition in Musical Audio Signals, each audio sample has an instrument and music genre attached, which lets us easily filter out the pop rock samples that we need.


### Networks (in completion)
  - [_WaveGAN_](https://arxiv.org/pdf/1802.04208.pdf)
  - [_SpecGAN_](https://arxiv.org/pdf/1802.04208.pdf)
  - [_GANSynth_](https://openreview.net/pdf?id=H1xQVn09FX)
  - [_MP3net_](https://arxiv.org/pdf/2101.04785.pdf)

## Deployment
&#9888; **BEWARE: COMMANDS WERE WRITTEN FOR MACOS/UNIX COMMAND LINE THUS WILL REQUIRE SOME FORMATTING FOR WINDOWS.** &#9888;

### Environment
We suggest creating a new conda environment to run this project and assume that you have anaconda installed on your machine.<br>
Use the following commands in your terminal from the project repository to create an environment with python version 3.6.8:
```
# create conda environment with specific python version
conda create -n api_gan python=3.6

# activate the conda environment
conda activate api_gan

# install the required packages
pip install -r requirements.txt
```

Once you are all set and have all the required packages installed, you are able to use all the scripts present in this project.

### Dataset
The data has already been trimmed and filtered, but it's source is still available in the above sections. Furthermore, the scripts used to preprocess the datasets are available in the data preparation folder. The scripts are available as python scripts or Jupyter Notebooks.<br>
For more information on the operations, use the following commands:
```
# for audio slicing
python data_preparation/audio_slicing.py -h
# for file filtering
python data_preparation/file_filtering.py -h
```

To run the pre-processing scripts on other datasets, run the following command in your terminal:
```
# for audio slicing
python data_preparation/audio_slicing.py -s 1 -dl /temp/data/location/ -el /temp/export/location/

# for file filtering
python data_preparation/file_filtering.py -dl /temp/data/location/ -el /temp/export/location/
```

### WaveGAN
The implementation of WaveGAN in this project is forked from the [WaveGAN](https://github.com/chrisdonahue/wavegan) repository developped by [Chris Donahue](https://github.com/chrisdonahue), [Andrés Marafioti](https://github.com/andimarafioti) and [Christian Clauss](https://github.com/cclauss).

To train (or resume training) a WaveGAN network, use the following command:

```
python wavegan/train_wavegan.py train ./path/to/save/location --data_dir ./path/to/data/ --data_first_slice --data_pad_end --data_prefetch_gpu_num -1 
```

This command is used to train on short data samples. To train on long audio samples (longer than a few seconds), remove `--data_first_slice` and `--data_pad_end` from the command.<br>
Breakdown:
  - `--data_dir` specifies the data used for training the GAN.
  - `--data_first_slice` is needed when training on shorter audio samples to only extract the first slice from each audio file.
  - `--data_pad_end` is needed when training on extremely short audio samples (less than 16384 samples each) so that they get zero padded to fill the slice.
  - `--data_prefetch_gpu_num -1` to train the GAN on the CPU.

While it should be possible to train WaveGAN on the GPU, we were unable to have this feature working. If you want to try setting up the code to handle GPU training (we won't cover how to enable you GPU for it), use the following command before training the network:

```
# for UNIX
export CUDA_VISIBLE_DEVICES=0
```
```
# for Windows
set CUDA_VISIBLE_DEVICES=0
```

The code also implements a function to create .wav samples when a checkpoint is created:

```
python wavegan/train_wavegan.py preview ./path/to/save/location
```

To back up checkpoints every hour (GAN training may occasionally collapse so it's good to have backups):
```
python wavegan/backup.py ./path/to/save/location 60
```

We have provided a notbook to generate audio in `generate.ipynb`, but we recommend using the following [colab notebook](https://colab.research.google.com/drive/18s5r2tCazWHMyVK-jGosullHOqUNd8I6?usp=sharing), as Google Colab provides easy access to GPUs.
<br><br>

### SpecGAN
The implementation of SpecGAN in this project is forked from the [WaveGAN](https://github.com/chrisdonahue/wavegan) repository developped by [Chris Donahue](https://github.com/chrisdonahue), [Andrés Marafioti](https://github.com/andimarafioti) and [Christian Clauss](https://github.com/cclauss).

Before training a SpecGAN network, it is necessary to compute mean and variance of each spectrogram bin to use for normalization:
```
python wavegan/train_specgan.py moments ./path/to/save/location --data_dir ./path/to/data/ --data_moments_fp ./path/to/save/location/moments.pkl
```

To train (or resume training) a SpecGAN network, use the following command:

```
python wavegan/train_specgan.py train ./path/to/save/location --data_dir ./path/to/data/ --data_first_slice --data_pad_end --data_prefetch_gpu_num -1 --data_moments_fp ./path/to/save/location/moments.cpkl
```

This command is used to train on short data samples. To train on long audio samples (longer than a few seconds), remove `--data_first_slice` and `--data_pad_end` from the command.<br>
Breakdown:
  - `--data_dir` specifies the data used for training the GAN.
  - `--data_first_slice` is needed when training on shorter audio samples to only extract the first slice from each audio file.
  - `--data_pad_end` is needed when training on extremely short audio samples (less than 16384 samples each) so that they get zero padded to fill the slice.
  - `--data_prefetch_gpu_num -1` to train the GAN on the CPU.

SpecGAN also required more memory when running on our system, thus we had to change the batch size to 16 (it is set to 32 by default) using `--train_batch_size 16`.

Similar to WaveGAN, the implementation provides a function to create .wav audio samples when a checkpoint of the model is created. To run this script, use the following command:
```
python wavegan/train_specgan.py preview ./path/to/save/location --data_moments_fp ./path/to/save/location/moments.cpkl
```

Similar to WaveGAN, you can back up checkpoints every hour with the following command:
```
python wavegan/backup.py ./path/to/save/location 60
```

The [colab notebook](https://colab.research.google.com/drive/18s5r2tCazWHMyVK-jGosullHOqUNd8I6?usp=sharing) also handles generating SpecGAN samples.
<br><br>

### GANSynth
add more
<br><br>

### MP3net
add more
