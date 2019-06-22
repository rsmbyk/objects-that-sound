import os
from operator import attrgetter

import click


class DataDir(click.Path):
    name = 'data_dir'
    DEFAULT_VALUE = 'data'

    def __init__(self):
        super().__init__(file_okay=False, resolve_path=True)

    def convert(self, value, param, ctx):
        value = super().convert(value, param, ctx)
        os.makedirs(value, exist_ok=True)
        return value


class DataDirDependent(click.Path):
    DEFAULT_VALUE = None

    def __init__(self, data_dir_option_name, file_okay=True, dir_okay=True):
        super().__init__(file_okay=file_okay, dir_okay=dir_okay)
        self.data_dir_option_name = data_dir_option_name

    def postpone(self, value, param, ctx):
        params = list(filter(lambda p: p.human_readable_name.lower() == 'DATA_DIR'.lower(),
                             ctx.command.params))
        if len(params) != 1:
            self.fail('DATA_DIR options is required to use {}'.format(param.human_readable_name),
                      param, ctx)

        data_dir_param = params[0]
        param_convert = data_dir_param.type.convert

        def convert_hook(*args):
            data_dir_param.type.convert = param_convert
            data_dir = data_dir_param.type.convert(*args)
            try:
                rv = self.real_convert(data_dir, value)
            except ValueError as e:
                self.fail(e.args, param, ctx)
            rv = super(DataDirDependent, self).convert(rv, param, ctx)
            ctx.params[param.human_readable_name] = rv
            return data_dir

        data_dir_param.type.convert = convert_hook

    def real_convert(self, data_dir, value):
        value = os.path.join(data_dir, self.get_subpath(value))
        if self.dir_okay:
            os.makedirs(value, exist_ok=True)
        return value

    def get_subpath(self, value):
        raise NotImplementedError()

    def convert(self, value, param, ctx):
        if value == self.DEFAULT_VALUE:
            if self.data_dir_option_name not in ctx.params:
                self.postpone(value, param, ctx)
                return value
            data_dir = ctx.params[self.data_dir_option_name]
            value = self.real_convert(data_dir, value)
        return super().convert(value, param, ctx)


class SegmentsFile(DataDirDependent):
    name = 'segments_file'
    DEFAULT_VALUE = 'balanced'

    def __init__(self, data_dir_option_name='data_dir'):
        super().__init__(data_dir_option_name, dir_okay=False)

    @property
    def available_segments(self):
        return (('balanced', 'balanced_train_segments.csv'),
                ('unbalanced', 'unbalanced_train_segments.csv'),
                ('eval', 'eval_segments.csv'))

    @property
    def segments_files(self):
        aliases, filenames = zip(*self.available_segments)

        def append_to_segments_dir(x):
            return os.path.join('segments', x)

        filenames = list(map(append_to_segments_dir, filenames))
        return {x[0]: x[1] for x in zip(aliases, filenames)}

    def get_metavar(self, param):
        return 'FILE [%s]' % '|'.join(self.segments_files)

    def real_convert(self, data_dir, value):
        value = super().real_convert(data_dir, value)
        if not os.path.exists(value):
            raise 'Segments file "{}" does not exists. Please init the dataset first.'.format(value)
        return value

    def get_subpath(self, value):
        return self.segments_files[value]

    def convert(self, value, param, ctx):
        if value in self.segments_files:
            if self.data_dir_option_name not in ctx.params:
                self.postpone(value, param, ctx)
                return value
            data_dir = ctx.params[self.data_dir_option_name]
            value = self.real_convert(data_dir, value)
        return super().convert(value, param, ctx)


class TrainSegmentsFile(SegmentsFile):
    name = 'train_segments_file'
    DEFAULT_VALUE = 'balanced'

    @property
    def available_segments(self):
        return (('balanced', 'balanced_train_segments.csv'),
                ('unbalanced', 'unbalanced_train_segments.csv'))


class ValidSegmentsFile(SegmentsFile):
    name = 'valid_segments_file'
    DEFAULT_VALUE = 'eval'

    @property
    def available_segments(self):
        return ('eval', 'eval_segments.csv'),


class OntologyFile(DataDirDependent):
    name = 'ontology_file'
    DEFAULT_VALUE = 'DATA_DIR/labels/ontology.json'

    def __init__(self, data_dir_option_name='data_dir'):
        super().__init__(data_dir_option_name, dir_okay=False)

    def real_convert(self, data_dir, value):
        value = super().real_convert(data_dir, value)
        if not os.path.exists(value):
            raise 'Ontology file "{}" does not exists. Please init the dataset first.'.format(value)
        return value

    def get_subpath(self, value):
        return os.path.join('labels', 'ontology.json')


class LogDir(DataDirDependent):
    name = 'logdir'
    DEFAULT_VALUE = 'DATA_DIR/logs'

    def __init__(self, data_dir_option_name='data_dir', output_option_name='output'):
        super().__init__(data_dir_option_name, file_okay=False)
        self.output_option_name = output_option_name

    def get_subpath(self, value):
        return 'logs'

    def convert(self, value, param, ctx):
        value = super().convert(value, param, ctx)
        print(list(map(attrgetter('human_readable_name'), ctx.command.params)))
        if self.output_option_name not in ctx.params:
            self.fail('Output options is required',
                      param, ctx)

        value = os.path.join(value, ctx.params[self.output_option_name])
        return value


class CheckpointsFilepath(DataDirDependent):
    name = 'checkpoints'
    DEFAULT_VALUE = 'DATA_DIR/checkpoints'

    def __init__(self, data_dir_option_name='data_dir', output_option_name='output'):
        super().__init__(data_dir_option_name, file_okay=False)
        self.output_option_name = output_option_name

    def get_subpath(self, value):
        return 'checkpoints'

    def convert(self, value, param, ctx):
        value = super().convert(value, param, ctx)

        if self.output_option_name not in ctx.params:
            self.fail('Output options is required',
                      param, ctx)

        filepath = '{epoch:03d}-{loss:03f}.h5'
        value = os.path.join(value, ctx.params[self.output_option_name], filepath)
        os.makedirs(os.path.dirname(value), exist_ok=True)
        return value


class ModelDir(DataDirDependent):
    name = 'modeldir'
    DEFAULT_VALUE = 'DATA_DIR/models'

    def __init__(self, data_dir_option_name='data_dir'):
        super().__init__(data_dir_option_name, file_okay=False)

    def get_subpath(self, value):
        return 'models'


class TempDir(DataDirDependent):
    name = 'tempdir'
    DEFAULT_VALUE = 'DATA_DIR/.temp'

    def __init__(self, data_dir_option_name='data_dir'):
        super().__init__(data_dir_option_name, file_okay=False)

    def get_subpath(self, value):
        return '.temp'
