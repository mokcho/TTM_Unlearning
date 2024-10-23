# TTM_Unlearning
Machine Unlearning in Text-to-Music Models

# Dataset

We will use MusicCaps, [Extended MusicCaps and MelBench](https://github.com/schowdhury671/melfusion/tree/main)

## Prepare MusicCaps Audio

This code obtains MusicCaps annotations from huggingface, and extracts corresponding audio files to ./data/MusicCaps/audio as default.
Change the audio path to your liking.

```
python ./preprocess/MusicCaps.py --audio_path {audio path}
```
