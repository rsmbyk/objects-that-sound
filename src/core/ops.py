import os

import pandas as pd
import tensorflow as tf

from util import ffmpeg, tensorplow as tp


def extract_frames(raw, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(output_dir):
        raise NotADirectoryError('OUTPUT_DIR ({})'.format(output_dir))

    ffmpeg.ffmpeg(raw, os.path.join(output_dir, '%d.jpg'),
                  r=25,
                  start_number=0,
                  vf='scale=256:256:force_original_aspect_ratio=increase')


def extract_audio(raw, output):
    if os.path.exists(output):
        return

    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

    ffmpeg.ffmpeg(raw, output,
                  ac=1,
                  ar=48000)


def get_video_duration(filename):
    output = ffmpeg.ffprobe(filename,
                            v='error',
                            show_entries='format=duration',
                            of='default=noprint_wrappers=1:nokey=1')
    return float(output.strip())


def compute_spectrogram(waveform, output):
    if os.path.exists(output):
        return

    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

    spc = tp.spectrogram(waveform,
                         sample_rate=48000,
                         window_length=0.01,
                         overlap=0.5)

    squeeze = tf.squeeze(spc, -1)
    df = pd.DataFrame(squeeze.numpy())
    df.to_csv(output, header=False, index=False)


def load_spectrogram(filename):
    csv = pd.read_csv(filename, header=None)
    return tf.expand_dims(csv.to_numpy(), -1)
