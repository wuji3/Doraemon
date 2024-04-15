import logging
from typing import Union
import yaml

class SmartLogger:

    _Instance = None
    _Flag = False

    def __new__(cls, *args, **kwargs):
        if cls._Instance is None:
            cls._Instance = super().__new__(cls)
        return cls._Instance

    def __init__(self, filename = None, level: int = 1):
        if not self.__class__._Flag:
            self.__class__._Flag = True
            # logger -> Singleton
            self.file_logger = self.create_logger('file', 'file', filename=filename) if filename is not None else None
            self.console_logger = self.create_logger('console', 'console')
            self.level = level

    def log(self, msg: Union[str, dict]):
        if isinstance(msg, dict):
            self.file_logger.info(yaml.dump(msg, sort_keys=False, default_flow_style=False))

        else: self.file_logger.info(msg)

    def console(self, msg: Union[str, dict]):
        if isinstance(msg, dict):
            self.console_logger.info('\n'+str(yaml.dump(msg, sort_keys=False, default_flow_style=False)))

        else:
            self.console_logger.info(msg)

    def both(self, msg: Union[str, dict]):
        self.log(msg)
        self.console(msg)

    def create_logger(self, name: str, kind: str, filename = None):
        assert kind in {'file', 'console'}
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        # format
        file_format = logging.Formatter(fmt='%(asctime)-20s%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if kind == 'file':
            handler = logging.FileHandler(filename=filename)
            handler.setFormatter(file_format)
            logger.addHandler(handler)
        elif kind == 'console':
            handler = logging.StreamHandler()
            handler.setFormatter(file_format)
            logger.addHandler(handler)
        return logger









