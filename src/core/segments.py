import glob
import json
import os
import random
from functools import reduce
from operator import itemgetter
from typing import Union, List

import math
import pandas as pd

from core import ops
from util import tensorplow as tp, youtube as yt


class Segment:
    attr_names = [
        'start_seconds',
        'end_seconds',
        'positive_labels',
        'duration',
        'wavelength'
    ]

    def __init__(self, root_dir, ytid, start_seconds, end_seconds, positive_labels):
        self.root_dir = root_dir
        self.ytid = ytid
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.positive_labels = positive_labels

        self.__duration = None
        self.__wavelength = None
        self.__positive_indices = None

        self.load_attributes()
        self.start_seconds = math.ceil(self.start_seconds)
        self.end_seconds = math.floor(self.end_seconds)

    def load_attributes(self):
        if os.path.exists(self.attrs_file):
            with open(self.attrs_file) as attrs_file:
                try:
                    attrs = json.load(attrs_file)
                except json.JSONDecodeError:
                    attrs = dict()

            for name in self.attr_names:
                if name in attrs:
                    setattr(self, '_Segment__' + name, attrs[name])

    def save_attribute(self, key, value):
        if key not in self.attr_names:
            raise ValueError('key {} can\'t be saved to attributes file'.format(key))

        if not os.path.exists(self.dir):
            return

        if not os.path.exists(self.attrs_file):
            with open(self.attrs_file, 'w'):
                # to make sure the file is created
                pass

        with open(self.attrs_file) as attrs_file:
            try:
                attrs = json.load(attrs_file)
            except json.JSONDecodeError:
                attrs = dict()

        if key not in attrs:
            attrs[key] = value

        with open(self.attrs_file, 'w') as attrs_file:
            json.dump(attrs, attrs_file, indent=4)

    @property
    def dir(self):
        return os.path.join(self.root_dir, self.ytid)

    @property
    def frames_dir(self):
        return os.path.join(self.dir, 'frames')

    @property
    def spectrograms_dir(self):
        return os.path.join(self.dir, 'spectrograms')

    @property
    def raw_prefix(self):
        return os.path.join(self.dir, self.ytid)

    @property
    def ydl_outtmpl(self):
        return '{}.%(ext)s'.format(self.raw_prefix)

    @property
    def raw(self):
        paths = map((self.raw_prefix + '.{}').format, yt.outfile_extensions)
        globs = map(glob.glob, paths)
        files = reduce(lambda a, b: a + b, globs)

        if not files:
            raise FileNotFoundError('RAW ({})'.format(self.ytid))
        if len(files) > 1:
            raise AssertionError('Multiple RAW files found: {}'.format(files))
        return files[0]

    @property
    def wav(self):
        return os.path.join(self.dir, '{}.wav'.format(self.ytid))

    @property
    def attrs_file(self):
        return os.path.join(self.dir, '{}.json'.format(self.ytid))

    def frame(self, index):
        return os.path.join(self.frames_dir, '{}.jpg'.format(index))

    def spectrogram(self, index):
        return os.path.join(self.spectrograms_dir, '{}.npz'.format(index))

    @property
    def frame_rate(self):
        # TODO: Do not use constant value
        return 25

    @property
    def sample_rate(self):
        # TODO: Do not use constant value
        return 48000

    @property
    def start_frames(self):
        return int(self.start_seconds * self.frame_rate)

    @property
    def end_frames(self):
        return int(self.end_seconds * self.frame_rate)

    @property
    def start_samples(self):
        return int(self.start_seconds * self.sample_rate)

    @property
    def end_samples(self):
        return int(self.end_seconds * self.sample_rate)

    def get_seconds(self, frame_index):
        return frame_index / self.frame_rate

    def get_sample_index(self, frame_index):
        return int(self.get_seconds(frame_index) * self.sample_rate)

    @property
    def positive_indices(self):
        if self.__positive_indices is None:
            if os.path.exists(self.frames_dir):
                frames = os.listdir(self.frames_dir)
                frames = map(lambda f: f.split('.'), frames)
                frames = map(itemgetter(0), frames)
                frames = map(int, frames)
                max_frame_index = max(frames)
            else:
                max_frame_index = len(self)
            self.__positive_indices = range(self.start_frames, min(self.end_frames, len(self), max_frame_index))
        return self.__positive_indices

    @property
    def duration(self):
        if self.__duration is None:
            self.__duration = ops.get_video_duration(self.raw)
        return self.__duration

    @property
    def wavelength(self):
        if self.__wavelength is None:
            self.__wavelength = len(self.waveform.audio)
        return self.__wavelength

    @property
    def waveform(self):
        if not os.path.exists(self.wav):
            ops.extract_audio(self.raw, self.wav)
        return tp.load_wav(self.wav)

    def load_frame(self, index):
        if not os.path.exists(self.frame(index)):
            ops.extract_frames(self.raw,
                               self.frames_dir,
                               self.start_seconds)
        return tp.load_image(self.frame(index))

    def load_spectrogram(self, index):
        if not os.path.exists(self.spectrogram(index)):
            start_index = self.get_sample_index(index)
            end_index = start_index + self.sample_rate
            waveform = self.waveform.audio[start_index:end_index]
            ops.compute_spectrogram(waveform, self.spectrogram(index))
        return ops.load_spectrogram(self.spectrogram(index))

    def get_positive_sample_index(self):
        frame_index = random.choice(self.positive_indices)

        audio_positive_indices = range(frame_index - self.frame_rate + 1, frame_index + 1)
        audio_index = max(0, random.choice(audio_positive_indices))

        return frame_index, audio_index

    def get_positive_sample(self):
        frame_index, audio_index = self.get_positive_sample_index()
        return self.load_frame(frame_index), self.load_spectrogram(audio_index)

    @property
    def is_available(self):
        try:
            return self.raw is not None
        except FileNotFoundError:
            return False

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        key = key.strip('_')
        if key in self.attr_names:
            self.save_attribute(key, value)

    def __eq__(self, other):
        if isinstance(other, Segment):
            return self.ytid == other.ytid
        if isinstance(other, str):
            return self.ytid == other
        return False

    def __len__(self):
        waveform_end = math.floor(self.wavelength / self.sample_rate)
        return (waveform_end - 1) * self.frame_rate


class SegmentsWrapper:
    def __init__(self, filename, root_dir):
        if not isinstance(filename, str):
            raise TypeError('FILENAME can\'t be of type {}'.format(type(filename).__name__))

        if not os.path.exists(filename):
            raise FileNotFoundError('FILENAME ({})'.format(filename))

        self.__filename = filename
        self.__root_dir = root_dir
        self.__segments = None
        self.__segments_dict = None

    @property
    def filename(self):
        return self.__filename

    @property
    def root_dir(self):
        return self.__root_dir

    @property
    def segments(self):
        if self.__segments is None:
            def to_segment(row):
                row = [*row[1]]
                row[-1] = row[-1].strip('"').split(',')
                return Segment(self.root_dir, *row)

            csv = pd.read_csv(self.filename,
                              sep=', ',
                              header=None,
                              engine='python',
                              comment='#')
            self.__segments = list(map(to_segment, csv.iterrows()))
        return self.__segments

    def to_dict(self):
        if self.__segments_dict is None:
            self.__segments_dict =\
                {segment.ytid: segment for segment in self.segments}
        return self.__segments_dict

    def filter(self, *labels: str):
        if not all(map(lambda x: isinstance(x, str), labels)):
            raise TypeError('Can only filter by str')

        new_segments = SegmentsWrapper('.', self.root_dir)

        # at least one positive label match the filter label
        new_segments.__segments =\
            list(filter(lambda s: any(map(lambda l: l in labels, s.positive_labels)), self.segments))
        return new_segments

    def __contains__(self, item):
        return item in self.to_dict()

    def __getitem__(self, item) -> Union[Segment, List[Segment]]:
        if isinstance(item, str):
            return self.to_dict()[item]
        if isinstance(item, (int, slice)):
            return self.segments[item]
        raise KeyError('Can only retrieve item by ytid or row index')

    def __len__(self):
        return len(self.segments)
