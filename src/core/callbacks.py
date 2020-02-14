import os
from collections import defaultdict

import numpy as np
from tensorflow.keras.callbacks import Callback


class NumpyzBoard(Callback):
    def __init__(self, logdir, period=1, resume_training=True):
        super().__init__()
        self.period = period
        self.logdir = logdir
        self.numpyboard_dir = os.path.join(logdir, 'numpyboard')
        self.batch_history_file = os.path.join(self.numpyboard_dir, 'batch.npz')
        self.epoch_history_file = os.path.join(self.numpyboard_dir, 'epoch.npz')
        self.batch_history = defaultdict(list)
        self.epoch_history = defaultdict(list)

        os.makedirs(self.numpyboard_dir, exist_ok=True)

        if resume_training:
            if os.path.exists(self.batch_history_file):
                batch = np.load(self.batch_history_file)
                for k in batch.keys():
                    self.batch_history[k] = list(batch[k])

            if os.path.exists(self.epoch_history_file):
                epoch = np.load(self.epoch_history_file)
                for k in epoch.keys():
                    self.epoch_history[k] = list(epoch[k])

    def on_batch_end(self, batch, logs=None):
        for l, val in logs.items():
            self.batch_history[l].append(val)

    def on_epoch_end(self, epoch, logs=None):
        for l, val in logs.items():
            self.epoch_history[l].append(val)

            for _ in range(5):
                self.batch_history[l].append(0)

        np.savez_compressed(self.batch_history_file + '.temp', **self.batch_history)
        np.savez_compressed(self.epoch_history_file + '.temp', **self.epoch_history)

        if epoch % self.period == 0:
            np.savez_compressed(self.batch_history_file, **self.batch_history)
            np.savez_compressed(self.epoch_history_file, **self.epoch_history)
