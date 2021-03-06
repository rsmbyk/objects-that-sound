import os

import numpy as np

from util import ffmpeg, tensorplow as tp


def extract_frames(raw, output_dir, start_time=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(output_dir):
        raise NotADirectoryError('OUTPUT_DIR ({})'.format(output_dir))

    ffmpeg.ffmpeg(raw, os.path.join(output_dir, '%d.jpg'),
                  r=25,
                  start_number=start_time * 25,
                  frames=250,
                  ss=start_time,
                  vf='scale=256:256:force_original_aspect_ratio=increase')


def extract_all_frames(raw, output_dir):
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

    float32 = spc.numpy().astype(np.float32)
    np.savez_compressed(output, spectrogram=float32)


def load_spectrogram(filename):
    npz = np.load(filename)
    spc = npz['spectrogram']
    return np.expand_dims(spc, -1)


def merge_frames(frames_dir, audio, output):
    if not os.path.exists(frames_dir):
        raise FileNotFoundError('FRAMES_DIR ({})'.format(frames_dir))

    if not os.path.isdir(frames_dir):
        raise NotADirectoryError('FRAMES_DIR ({})'.format(frames_dir))

    if not os.path.exists(audio):
        raise FileNotFoundError('AUDIO ({})'.format(audio))

    if os.path.exists(output):
        return

    ffmpeg.ffmpeg(audio, output,
                  i=os.path.join(frames_dir, '%d.png'),
                  r=25,
                  pix_fmt='yuv420p',
                  **{'c:v': 'libvpx'})
