from cli import groups, options, arguments, utils
from core import commands


@groups.dataset.command()
@arguments.data_dir
@options.overwrite
@utils.display_params
def init(**kwargs):
    """ Download all the dataset files. """
    commands.dataset.init(**kwargs)


@groups.dataset.command()
@options.data_dir
@options.segments
@options.ontology
@options.blacklist
@options.seed
@options.limit
@options.min_size
@options.max_size
@arguments.labels
@utils.display_params
def download(**kwargs):
    """ Download audioset segments. """
    commands.dataset.download(**kwargs)


@groups.dataset.command()
@arguments.data_dir
@options.segments
@options.workers
@utils.display_params
def preprocess(**kwargs):
    """ Preprocess the dataset. """
    commands.dataset.preprocess(**kwargs)


@groups.dataset.command()
@arguments.data_dir
@options.segments
@options.remove_audio
@options.remove_frames
@options.remove_spectrograms
@utils.display_params
def cleanup(**kwargs):
    """ Cleanup the dataset. """
    commands.dataset.cleanup(**kwargs)


@groups.dataset.command()
@options.data_dir
@options.segments
@options.ontology
@arguments.labels
@arguments.output_file
@utils.display_params
def compress_segments(**kwargs):
    """ Compress currently available segments into a new segments file. """
    commands.dataset.compress_segments(**kwargs)
