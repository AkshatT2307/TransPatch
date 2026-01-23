import yaml
from collections import namedtuple
import numpy as np
import math
import torch
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation')
from utils.dirs import create_dirs
import datetime
import os
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir, level: str = "INFO", minimal_banner: bool = False):
  log_file_format = "%(message)s"
  log_console_format = "%(message)s"

  # Resolve level string to logging constant
  level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
  }
  console_level = level_map.get(str(level).upper(), logging.INFO)

  # Main logger
  main_logger = logging.getLogger()
  if main_logger.hasHandlers():
    main_logger.handlers.clear()
  main_logger.setLevel(console_level)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(console_level)
  console_handler.setFormatter(Formatter(log_console_format))

  # Always capture DEBUG to file, regardless of console level
  exp_file_handler = RotatingFileHandler(os.path.join(log_dir, 'exp_debug.log'), maxBytes=10**16, backupCount=5)
  exp_file_handler.setLevel(logging.DEBUG)
  exp_file_handler.setFormatter(Formatter(log_file_format))

  main_logger.addHandler(console_handler)
  main_logger.addHandler(exp_file_handler)

  # Optional minimal banner flag stored on logger for later checks
  setattr(main_logger, "_minimal_banner", bool(minimal_banner))

  return main_logger



def get_config_from_yaml(root,model = None):
  """
  Get the config from yaml file
  :param yaml_file: the path of the config file
  """

  # parse the configurations from the config json file provided
  with open(root, 'r') as config_file:
    try:
      config_dict = yaml.safe_load(config_file)
      config_dict['experiment']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
      if model is not None:
        config_dict['model']['name'] = model
      config = {}
      for i in config_dict:
        config[i] = namedtuple('Struct', config_dict[i].keys())(*config_dict[i].values())
      config = namedtuple('Struct', config.keys())(*config.values())
      return config
    except ValueError:
      print("INVALID YAML file format.. Please provide a good yaml file")
      exit(-1)




def process_config(config):
    """
    Get the config file from yaml file
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """

    # create some important directories to be used for that experiment.
    patch_dir = os.path.join(config.experiment.log_patch_address, "patch/")
    log_dir = os.path.join(config.experiment.log_patch_address, "logs/")
    create_dirs([patch_dir, log_dir])

    # setup logging in the project (configurable level + banner)
    log_level = getattr(config.experiment, 'log_level', 'INFO')
    minimal_banner = getattr(config.experiment, 'minimal_logs', False)
    main_logger = setup_logging(log_dir, level=log_level, minimal_banner=minimal_banner)

    # making sure that you have provided the exp_name.
    try:
      if not getattr(main_logger, "_minimal_banner", False):
        main_logger.info(" ****************************************************************************** ")
        main_logger.info(f'                  {config.experiment.name} at {str(datetime.datetime.now())}')
        main_logger.info(" ****************************************************************************** ")
      else:
        main_logger.info(f"{config.experiment.name} | {str(datetime.datetime.now())}")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in yaml file..")
        exit(-1)

    if not getattr(main_logger, "_minimal_banner", False):
      main_logger.info("Hi, This is root.")
      main_logger.info("After the configurations are successfully processed and dirs are created.")
      main_logger.info("The pipeline of the project will begin now.")


    # configuration = '\n'
    # for i in config:
    #     configuration += i+': '+str(config[i])+'\n'

    if not getattr(main_logger, "_minimal_banner", False):
      main_logger.info(log_config(config))

    return main_logger


def print_config(config_dict):
  for field in config_dict._fields:
    print(f'----------------- {field}--------------------')
    f = getattr(config_dict, field)
    for i in f._fields:
      print(i,':',getattr(f,i))


def log_config(config_dict):
  string = '------------- :: Config for this run :: -----------------\n'
  for field in config_dict._fields:
    string = string + f'----------------- {field}--------------------\n'
    f = getattr(config_dict, field)
    for i in f._fields:
      string += f'{i}:{getattr(f,i)}\n'
  return string
    












