from contextlib import contextmanager

from distill.logger import logger


@contextmanager
def exception_handler(*args, **kwargs):
    try:
        yield
    except Exception as err:
        logger.exception(err)
