import click

from cli import types

verbose =\
    click.option('-v', '--verbose', 'level',
                 flag_value=1,
                 help='be more verbose.')

quiet =\
    click.option('-q', '--quiet', 'level',
                 flag_value=-1,
                 help='be more quiet.')

prompt_yes =\
    click.option('-y', '--yes', 'prompt',
                 flag_value=1,
                 help='answer all prompts as yes')

prompt_no =\
    click.option('-n', '--no', 'prompt',
                 flag_value=-1,
                 help='answer all prompts as no')

yes =\
    click.option('-y', '--yes',
                 is_flag=True,
                 help='answer all prompts as yes.')

no =\
    click.option('-n', '--no',
                 is_flag=True,
                 help='answer all prompts as no.')

overwrite =\
    click.option('-o', '--overwrite',
                 is_flag=True,
                 help='overwrite all files.')

data_dir =\
    click.option('-d', '--data_dir',
                 type=types.DataDir(),
                 default=types.DataDir.DEFAULT_VALUE,
                 show_default=True,
                 help='data directory to use.')

segments =\
    click.option('-s', '--segments',
                 type=types.SegmentsFile(),
                 default=types.SegmentsFile.DEFAULT_VALUE,
                 show_default=True,
                 help='segments file to use.')

ontology =\
    click.option('-o', '--ontology',
                 type=types.OntologyFile(),
                 default=types.OntologyFile.DEFAULT_VALUE,
                 show_default=True,
                 help='ontology file to use')

limit =\
    click.option('-l', '--limit',
                 type=click.IntRange(min=0),
                 help='limit number of segments for each label to LIMIT (0 for no limit).')

min_size =\
    click.option('-mi', '--min-size',
                 type=click.IntRange(min=0),
                 help='minimum allowed filesize to be downloaded (in MB).')

max_size =\
    click.option('-ma', '--max-size',
                 type=click.IntRange(min=0),
                 help='maximum allowed filesize to be downloaded (in MB).')

blacklist =\
    click.option('-b', '--blacklist',
                 type=click.Path(exists=True, dir_okay=False),
                 help='list of segments id to be ignored.')

workers =\
    click.option('-w', '--workers',
                 type=click.IntRange(min=0),
                 default=1,
                 show_default=True,
                 help='number of workers/thread to use.')

preprocess =\
    click.option('--preprocess',
                 is_flag=True,
                 help='preprocess the video upon download success.')

seed =\
    click.option('-a', '--seed',
                 type=click.INT,
                 help='seed value to use.')

network =\
    click.option('-n', '--network',
                 type=click.Choice(['l3', 'ave', 'avol']),
                 default='avol',
                 show_default=True,
                 help='AVC architecture to use.')

resume_training =\
    click.option('-rt', '--resume-training',
                 is_flag=True,
                 help='resume previous training')

train_segments =\
    click.option('-ts', '--train-segments',
                 type=types.TrainSegmentsFile(),
                 default=types.TrainSegmentsFile.DEFAULT_VALUE,
                 help='segments file to use as training data.')

valid_segments =\
    click.option('-vs', '--valid-segments',
                 type=types.ValidSegmentsFile(),
                 default=types.ValidSegmentsFile.DEFAULT_VALUE,
                 help='segments file to use as validation data.')

negative_segments =\
    click.option('-ns', '--negative-segments',
                 type=types.TrainSegmentsFile(),
                 default=types.TrainSegmentsFile.DEFAULT_VALUE,
                 help='segments file to use as negative data.')

epochs =\
    click.option('-e', '--epochs',
                 type=click.IntRange(min=1),
                 default=1,
                 show_default=True,
                 help='number of epochs to train.')

initial_epoch =\
    click.option('-ie', '--initial-epoch',
                 type=click.IntRange(min=0),
                 default=0,
                 show_default=True,
                 help='initial epoch to use.')

checkpoints_period =\
    click.option('-cpp', '--checkpoints-period',
                 type=click.IntRange(min=1),
                 default=1,
                 show_default=True,
                 help='checkpoints period.')

logdir =\
    click.option('-ld', '--logdir',
                 type=types.LogDir(),
                 default=types.LogDir.DEFAULT_VALUE,
                 show_default=True,
                 help='logdir for TensorBoard.')

checkpoints =\
    click.option('-cp', '--checkpoints',
                 type=types.CheckpointsFilepath(),
                 default=types.CheckpointsFilepath.DEFAULT_VALUE,
                 show_default=True,
                 help='directory to store model checkpoints.')

modeldir =\
    click.option('-md', '--modeldir',
                 type=types.ModelDir(),
                 default=types.ModelDir.DEFAULT_VALUE,
                 show_default=True,
                 help='directory to save the final model.')

threshold =\
    click.option('-th', '--threshold',
                 type=click.FloatRange(min=0., max=1.),
                 default=0.5,
                 show_default=True,
                 help='threshold value to use.')

tempdir =\
    click.option('-td', '--tempdir',
                 type=types.TempDir(),
                 default=types.TempDir.DEFAULT_VALUE,
                 show_default=True,
                 help='temp directory to use.')

output =\
    click.option('-out', '--output',
                 type=click.STRING,
                 help='output name')
