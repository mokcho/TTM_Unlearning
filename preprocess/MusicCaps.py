import os
import csv
import argparse
import subprocess
from pathlib import Path
from datasets import load_dataset, Audio
from yt_dlp import YoutubeDL
from tqdm import tqdm

def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='./tmp/musiccaps',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    command = f"""
        yt-dlp --quiet --force-keyframes-at-cuts --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def main(
    data_dir: str,
    sampling_rate: int = 44100,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    ds = load_dataset('google/MusicCaps', split='train')
    
    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )

        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    return ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    ).cast_column('audio', Audio(sampling_rate=sampling_rate))

        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="./data/MusicCaps/audio")
    #parser.add_argument("--annotation_path", type=str, default="./data/MusicCaps/annotation")
    args = parser.parse_args()
    
    # get annotated csv from Huggingface
    os.makedirs(args.audio_path, exist_ok=True)
    
    ds = main(args.audio_path, num_proc=16, limit=32)
    