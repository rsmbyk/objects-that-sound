import os
import stat

import wget


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
