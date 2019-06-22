from cli import options, groups, arguments, utils
from core import commands


@groups.avc.command()
@arguments.labels
@options.data_dir
@options.train_segments
@options.valid_segments
@options.negative_segments
@options.ontology
@options.seed
@options.network
@options.resume_training
@options.epochs
@options.initial_epoch
@options.checkpoints_period
@options.logdir
@options.checkpoints
@options.modeldir
@options.output
@utils.display_params
def train(*args, **kwargs):
    commands.avc.train(*args, **kwargs)


@groups.avc.command()
@arguments.input_file
@arguments.model
@arguments.output_file
@options.data_dir
@options.tempdir
@options.workers
@options.threshold
@utils.display_params
def test(*args, **kwargs):
    commands.avc.test(*args, **kwargs)
