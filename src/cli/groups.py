import click


@click.group()
def ots():
    """ `Objects that Sound` by Google Deepmind """


@ots.group()
def dataset():
    """ Manage dataset """


@ots.group()
def avc():
    """ Execute AVC Tasks """
