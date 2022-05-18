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
