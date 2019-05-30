import os
import pprint
import stat
from operator import attrgetter, itemgetter

import pandas as pd
import wget

import util.youtube as yt
from core.ontology import Ontology
from core.segments import SegmentsWrapper


def init(data_dir, overwrite=False):
    dataset_files = {
        'assessments': [
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv',
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/rerated_video_ids.txt'
        ],
        'labels': [
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv',
            'https://raw.githubusercontent.com/audioset/ontology/master/ontology.json'
        ],
        'segments': [
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv',
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv',
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'
        ]
    }

    for parent_dir, urls in dataset_files.items():
        print('\n', parent_dir, sep='')

        parent_dir = os.path.join(data_dir, parent_dir)
        os.makedirs(parent_dir, exist_ok=True)

        files = list(map(lambda x: x.split('/')[-1], urls))

        for file, url in zip(files, urls):
            filename = os.path.join(parent_dir, file)
            print(filename)

            if os.path.exists(filename):
                print('Exists.', end=' ')

                if not overwrite:
                    print('No overwrite.')
                    continue

                print('Overwriting.')
                os.chmod(filename, stat.S_IWUSR | stat.S_IREAD)
                os.remove(filename)

            wget.download(url, filename)
            os.chmod(filename, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            print()

        print('Cleaning up {}.'.format(os.path.basename(parent_dir)))

        for file in os.listdir(parent_dir):
            if file not in files:
                os.remove(os.path.join(parent_dir, file))


def download(labels, data_dir, segments, ontology, limit=None, min_size=None, max_size=None, blacklist=None):
    segments = SegmentsWrapper(segments, os.path.join(data_dir, 'raw'))
    ontology = Ontology(ontology, os.path.join(data_dir, 'videos'))

    if blacklist is None:
        blacklist = pd.DataFrame(columns=['YTID', 'reason'])
    else:
        blacklist = pd.read_csv(blacklist)

    def segment_in_ontology(o):
        def decorator(s):
            return any(map(o.__contains__, s.positive_labels))
        return decorator

    def filter_by_ontology(s):
        def decorator(o):
            return list(filter(segment_in_ontology(o), s))
        return decorator

    ontologies = ontology.retrieve(*labels)
    segments = list(filter(segment_in_ontology(ontologies), segments))

    ontologies = list(map(ontology.retrieve, labels))
    downloaded = list(filter(attrgetter('is_available'), segments))
    downloaded = map(filter_by_ontology(downloaded), ontologies)
    downloaded = zip(map(attrgetter('name'), ontologies), downloaded)

    counter = {name: s for name, s in downloaded}
    pprint.pprint({name: len(s) for name, s in counter.items()})

    for segment in segments:
        finished = limit is not None and all(map(limit.__le__, map(len, counter.values())))
        print(list(map(len, counter.values())))

        if finished:
            break

        if not any(map(lambda x: segment.ytid in x, counter.values())):

            if limit is not None:
                ok = True

                for ont in ontologies:
                    if segment_in_ontology(ont)(segment) and len(counter[ont.name]) >= limit:
                        print('[{}] "{}" has reached limit.'.format(segment.ytid, ont.proper_name))
                        ok = False

                if not ok:
                    continue

            blacklisted = blacklist[blacklist['YTID'] == segment.ytid]
            if not blacklisted.empty:
                print('[{}] is blacklisted. {}.'.format(*blacklisted.values[0]))
                continue

            info = yt.info(segment.ytid)
            if info == -1:
                continue

            formats = filter(lambda x: 'filesize' in x, info['formats'])
            filesizes = map(itemgetter('filesize'), formats)
            filesizes = filter(lambda x: x is not None, filesizes)
            filesize = int(max(filesizes) / 1024 / 1024)

            if min_size is not None and filesize < min_size:
                print('[{}] smaller than min_size ({} MiB).'.format(segment.ytid, filesize))
                continue

            if max_size is not None and filesize > max_size:
                print('[{}] exceeds max_size ({} MiB).'.format(segment.ytid, filesize))
                continue

            yt.dl(segment.ytid, outtmpl=segment.ydl_outtmpl)

            for ont in ontologies:
                if segment_in_ontology(ont)(segment):
                    counter[ont.name].append(segment)
