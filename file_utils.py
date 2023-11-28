import logging
import tarfile
from pathlib import Path

import yaml

_TAR_BUFFER_SIZE = 100_000


def load_yaml(file_path):
    with open(file_path, "r") as file:
        configs = yaml.safe_load(file)
    return configs


def download_file(url: str, path: Path):
    """
    Download a file from ``url``.

    Arguments:
        url (str): URL to download the file from
        path (Path): The location to save the downloaded file
    """
    if path.exists():
        return

    import urllib.request

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    urllib.request.urlretrieve(url, path)


def _extract_tar_file(tar: tarfile.TarFile, f: tarfile.TarInfo, path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with tar.extractfile(f) as ef:
        with open(str(path), "wb") as o:
            while True:
                r = ef.read(_TAR_BUFFER_SIZE)
                if r:
                    o.write(r)
                else:
                    break


def extract_tar(tar_file: Path, to_path: Path):
    """
    Extract a ``.tar.gz`` file.

    Arguments:
        tar_file (Path): ``.tar.gz`` file
        to_path (Path): location to extract the contents
    """
    with tarfile.open(str(tar_file), "r:gz") as tar:
        files = tar.getmembers()
        for f in files:
            if f.isdir():
                pass
            elif f.isfile():
                _extract_tar_file(tar, f, to_path / f.name)
            else:
                logging.error(f"Unknown file type {f.name}")
