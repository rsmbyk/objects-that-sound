import click

from cli import types

labels =\
    click.argument('labels', nargs=-1, required=True)


data_dir =\
    click.argument('data_dir',
                   type=types.DataDir(),
                   default=types.DataDir.DEFAULT_VALUE)

input_file =\
    click.argument('input_file',
                   nargs=1,
                   required=True,
                   type=click.Path(exists=True, dir_okay=False, resolve_path=True))

model =\
    click.argument('model',
                   nargs=1,
                   required=True,
                   type=click.Path(exists=True, dir_okay=False, resolve_path=True))

output_file =\
    click.argument('output_file',
                   nargs=1,
                   required=True,
                   type=click.Path(dir_okay=False, resolve_path=True))
