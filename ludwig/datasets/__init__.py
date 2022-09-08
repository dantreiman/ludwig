import argparse
import importlib
import os
import shutil
from functools import cache
from typing import Dict, List

import yaml

from ludwig.datasets import configs
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import print_ludwig


def _load_dataset_config(config_filename: str):
    """Loads a dataset config."""
    config_path = os.path.join(os.path.dirname(configs.__file__), config_filename)
    with open(config_path) as f:
        return DatasetConfig(**yaml.safe_load(f))


@cache
def _get_dataset_configs() -> Dict[str, DatasetConfig]:
    """Returns all dataset configs indexed by name."""
    import importlib.resources

    config_files = [f for f in importlib.resources.contents(configs) if f.endswith(".yaml")]
    config_objects = [_load_dataset_config(f) for f in config_files]
    return {c.name: c for c in config_objects}


def _instantiate_dataset_loader(dataset_name):
    """Instantiates the dataset loader for a given dataset."""
    config = _get_dataset_configs()[dataset_name]
    class_name = config.loader.split(".")[-1]
    module_name = "." + ".".join(config.loader.split(".")[:-1])
    loader_module = importlib.import_module(module_name, package="ludwig.datasets.loaders")
    loader_cls = getattr(loader_module, class_name)
    return loader_cls(config)


def list_datasets() -> List[str]:
    return sorted(_get_dataset_configs().keys())


def describe_dataset(dataset_name: str) -> str:
    return _get_dataset_configs[dataset_name].description


def download_dataset(dataset_name: str, output_dir: str = "."):
    dataset = _instantiate_dataset_loader(dataset_name)
    _ = dataset.load()
    processed_path = dataset.processed_dataset_dir
    _copytree(processed_path, output_dir, dirs_exist_ok=True)


def _copytree(src, dst, symlinks=False, ignore=None, dirs_exist_ok=False):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore, dirs_exist_ok=dirs_exist_ok)
        else:
            shutil.copy2(s, d)


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This command downloads and lists Ludwig-ready datasets.",
        prog="ludwig datasets",
        usage="%(prog)s [options]",
    )
    sub_parsers = parser.add_subparsers(dest="command", help="download and list datasets")

    parser_download = sub_parsers.add_parser("download", help="download a dataset")
    parser_download.add_argument("dataset", help="dataset to download")
    parser_download.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="output directory to download into",
        required=False,
    )

    sub_parsers.add_parser("list", help="list datasets")

    parser_describe = sub_parsers.add_parser("describe", help="describe datasets")
    parser_describe.add_argument("dataset", help="dataset to describe")

    args = parser.parse_args(sys_argv)
    print_ludwig(f"Datasets {args.command}", LUDWIG_VERSION)

    if args.command == "list":
        datasets = list_datasets()
        for ds in datasets:
            print(ds)
    elif args.command == "describe":
        print(describe_dataset(args.dataset))
    elif args.command == "download":
        download_dataset(args.dataset, args.output_dir)
    else:
        raise ValueError(f"Unrecognized command: {args.command}")
