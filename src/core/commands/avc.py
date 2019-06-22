import random
import shutil
import threading
from operator import attrgetter

import cv2
import math
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras import Model, models as keras_models
from tensorflow.python.keras.callbacks import *

from core import models, ops
from core.callbacks import NumpyzBoard
from core.generator import SegmentsGenerator
from core.ontology import Ontology
from core.segments import SegmentsWrapper, Segment


def train(data_dir, train_segments, negative_segments, valid_segments, ontology,
          labels, seed, network, resume_training,
          epochs, initial_epoch, checkpoints_period,
          logdir, checkpoints, modeldir, output):
    random.seed(seed)
    tf.random.set_seed(seed)

    raw_dir = os.path.join(data_dir, 'raw')
    train_segments = SegmentsWrapper(train_segments, raw_dir)
    valid_segments = SegmentsWrapper(valid_segments, raw_dir)
    negative_segments = SegmentsWrapper(negative_segments, raw_dir)

    def segment_in_ontology(o):
        def decorator(s):
            return any(map(o.__contains__, s.positive_labels))

        return decorator

    videos_dir = os.path.join(data_dir, 'videos')
    ontology = Ontology(ontology, videos_dir)
    ontologies = ontology.retrieve(*labels)

    train_segments = filter(segment_in_ontology(ontologies), train_segments)
    train_segments = list(filter(attrgetter('is_available'), train_segments))

    valid_segments = filter(segment_in_ontology(ontologies), valid_segments)
    valid_segments = list(filter(attrgetter('is_available'), valid_segments))

    negative_segments = filter(segment_in_ontology(ontologies), negative_segments)
    negative_segments = list(filter(attrgetter('is_available'), negative_segments))

    os.makedirs(logdir, exist_ok=True)

    with open(os.path.join(logdir, 'train_segments.txt'), 'w') as outfile:
        outfile.writelines(list(map(attrgetter('ytid'), train_segments)))

    with open(os.path.join(logdir, 'valid_segments.txt'), 'w') as outfile:
        outfile.writelines(list(map(attrgetter('ytid'), valid_segments)))

    print(len(train_segments), len(valid_segments))
    model = models.retrieve_model(network)()

    train_generator = SegmentsGenerator(train_segments, negative_segments, model, 55)
    valid_generator = SegmentsGenerator(valid_segments, negative_segments, model, 34)

    def decayer(epoch):
        return 1e-5 * math.pow((94. / 100), ((1 + epoch) // 16))

    numpyz_board = NumpyzBoard(logdir,
                               period=checkpoints_period,
                               resume_training=resume_training)
    model_checkpoint = ModelCheckpoint(checkpoints,
                                       period=checkpoints_period)
    lr_scheduler = LearningRateScheduler(decayer)

    callbacks = [numpyz_board, model_checkpoint, lr_scheduler]

    if resume_training:
        checkpoints_dir = os.path.dirname(checkpoints)
        checkpoint_models = os.listdir(checkpoints_dir)
        checkpoint_models = {int(x.split('-')[0]): x for x in checkpoint_models}
        initial_epoch = max(checkpoint_models.keys())
        latest_model = checkpoint_models[initial_epoch]
        model = keras_models.load_model(os.path.join(checkpoints_dir, latest_model))
    else:
        model: Model = model.compile()

    model.fit_generator(train_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=34,
                        max_queue_size=21,
                        initial_epoch=initial_epoch)

    model_filepath = os.path.join(modeldir, '{}.h5'.format(output))
    model.save(model_filepath)
    print('Model save to', model_filepath)


def generate_color_heatmaps():
    list_of_colors = [(30, 198, 244), (99, 200, 72), (120, 50, 80), (200, 90, 140)]
    no_steps = 100
    final_color_heatmaps = list()

    def lerp_colour(c1, c2, t):
        return c1[0] + (c2[0] - c1[0]) * t, c1[1] + (c2[1] - c1[1]) * t, c1[2] + (c2[2] - c1[2]) * t

    for i in range(len(list_of_colors) - 2):
        for j in range(no_steps):
            final_color_heatmaps.append(lerp_colour(list_of_colors[i], list_of_colors[i + 1], j / no_steps))

    return final_color_heatmaps


# noinspection PyUnusedLocal
def test(input_file, model, output_file, data_dir, tempdir, threshold, workers):
    segment_dir = os.path.join(tempdir, os.path.splitext(os.path.basename(output_file))[0])
    os.makedirs(segment_dir, exist_ok=True)
    segment_id = os.path.basename(segment_dir)
    segment = Segment(os.path.dirname(segment_dir), segment_id, -1, -1, [])
    filename, ext = os.path.splitext(input_file)
    shutil.copyfile(input_file, os.path.join(segment_dir, segment_id + ext))

    outputs_dir = os.path.join(segment_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    data_output_dir = os.path.join(segment_dir, 'localization_outputs')
    os.makedirs(data_output_dir, exist_ok=True)

    model = keras_models.load_model(model)
    ops.extract_all_frames(segment.raw, segment.frames_dir)

    color_heatmaps = generate_color_heatmaps()
    failed_frames = list()

    def thread_function(thread_id, start_frame, end_frame):
        localization_output = K.function([model.input], [model.get_layer('sigmoid').output])
        for fi in range(start_frame, end_frame):
            outframe = os.path.join(outputs_dir, '{}.png'.format(fi))
            if not os.path.exists(outframe):
                try:
                    frame = segment.load_frame(fi)
                    spectrogram = segment.load_spectrogram(fi)
                except:
                    failed_frames.append(fi)
                    continue

                frame = cv2.resize(frame, (224, 224))
                spectrogram = np.expand_dims(cv2.resize(spectrogram, (200, 257)), -1)

                frame = np.expand_dims(frame / 255., 0)
                spectrogram = np.expand_dims(spectrogram, 0)
                outputs = localization_output([frame, spectrogram])

                output = outputs[0][0]
                np.savez_compressed(os.path.join(data_output_dir, '{}.npz'.format(fi)),
                                    out=output)
                max_val = np.max(output)

                if max_val > 0.5:
                    print('[Thread {}]'.format(thread_id),
                          'Processing frame', fi, 'of', end_frame,
                          '({})'.format(max_val), 'correspond')
                else:
                    print('[Thread {}]'.format(thread_id),
                          'Processing frame', fi, 'of', end_frame,
                          '({})'.format(max_val))

                heatmap = np.zeros((224, 224, 4))

                for i in range(14):
                    for j in range(14):
                        # value = color_heatmaps[int(output[i, j] * 100)]
                        value = output[i, j]
                        value = 0 if value < threshold else value
                        ii = i * 16
                        jj = j * 16
                        heatmap[ii:ii + 16, jj:jj + 16, 0] = 0.5
                        heatmap[ii:ii + 16, jj:jj + 16, 1] = 1
                        heatmap[ii:ii + 16, jj:jj + 16, 2] = 1
                        heatmap[ii:ii + 16, jj:jj + 16, 3] = value

                heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
                frame = Image.fromarray((frame[0] * 255.).astype(np.uint8))

                frame.paste(heatmap, (0, 0), heatmap)
                frame.save(outframe)
                # frame.save(os.path.join(data_output_dir, '{}.png'.format(fi)))

    threads = list()

    for idx in range(workers):
        thread_size = math.ceil(len(segment) / workers)
        thread_start = idx * thread_size
        thread_args = idx, thread_start, thread_start + thread_size
        thread = threading.Thread(target=thread_function, args=thread_args)
        threads.append(thread)
        thread.start()

    for idx, thread in enumerate(threads):
        thread.join()

    if len(failed_frames) > 0:
        print('The following frames cannot be processed:', failed_frames)
    else:
        ops.merge_frames(outputs_dir, segment.wav, output_file)
        print('Output saved to', output_file)
