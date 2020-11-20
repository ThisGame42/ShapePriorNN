import shutil
import pathlib
import glob
import os

from typing import List


class file_helper(object):
    """
        A normal file helper util class.
    """

    @staticmethod
    def bulk_move_to(files: list, dest_dir: str) -> None:
        for f in files:
            file_helper._move_to(f, dest_dir)

    @staticmethod
    def _move_to(f: str, dest_dir: str) -> None:
        if not os.path.exists(dest_dir):
            pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
        f_name = os.path.basename(f)
        abs_dir = os.path.join(dest_dir, f_name)
        shutil.move(f, abs_dir)

    @staticmethod
    def make_dir_if_none(tar_dir: str) -> None:
        if not os.path.exists(tar_dir):
            os.mkdir(tar_dir)

    @staticmethod
    def get_parent_dir(tar_dir: str) -> str:
        return os.path.abspath(os.path.join(tar_dir, os.pardir))

    @staticmethod
    def files_by_ext(tar_dir: str, ext: str) -> List[str]:
        files = glob.glob(os.path.join(tar_dir, f'*.{ext}'))
        files.sort()
        return files
