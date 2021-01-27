# -*- coding: utf-8 -*-

import click
from graph_model import GraphModel


@click.command()
@click.option("--verbose", "-v", default=2, help="verbose: 0 1 2 [default 2]")
@click.argument('config_path', metavar='<config_path>')
def main(config_path, verbose):
    config = GraphModel.load_config(config_path)
    clf = GraphModel(is_training=True, **config)
    clf.train(verbose=verbose)


if __name__ == "__main__":
    main()