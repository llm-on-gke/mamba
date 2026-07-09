# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

import logging
from contextlib import contextmanager

def get_logger():
  logger = logging.getLogger("parallel_playground")
  logger.setLevel(logging.INFO)
  if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
  return logger

def rank_log(rank, logger, message):
  if rank == 0:
    logger.info(message)
  else:
    logger.debug(f"[Rank {rank}] {message}")

@contextmanager
def profiler_range(name):
  yield

def profiler_range_push(name):
  pass

def profiler_range_pop():
  pass
