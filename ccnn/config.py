import os
from configparser import RawConfigParser

def get_config():

    config = RawConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

    return config