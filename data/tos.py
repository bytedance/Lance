# coding: utf-8

class TosWrapper:
    """Placeholder for removed remote object storage support.

    The public training framework only supports local parquet files with media
    bytes embedded in each row. Remote TOS access has been intentionally
    removed from the default code path.
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _raise_removed():
        raise NotImplementedError(
            "Remote TOS access is not supported. Use local parquet files with embedded image/video bytes."
        )

    def get_obj(self, *args, **kwargs):
        self._raise_removed()

    def get_obj_by_url(self, *args, **kwargs):
        self._raise_removed()

    def get_obj_meta_by_url(self, *args, **kwargs):
        self._raise_removed()

    def put_obj(self, *args, **kwargs):
        self._raise_removed()

__all__ = ["TosWrapper"]
