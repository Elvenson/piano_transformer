# Copyright 2019 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# Modification copyright 2020 Bui Quoc Bao
# Change notebook script into package

import numpy as np
import os
import tensorflow as tf

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

import magenta.music as mm
from magenta.models.score2perf import score2perf

logging = tf.compat.v1.logging


class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
	@property
	def add_eos_symbol(self):
		return True


def decode(ids, encoder):
	"""Decode a list of IDs."""
	ids = list(ids)
	if text_encoder.EOS_ID in ids:
		ids = ids[:ids.index(text_encoder.EOS_ID)]
	return encoder.decode(ids)


def unconditional_input_generator(targets, decode_length):
	"""Estimator input function for unconditional Transformer."""
	while True:
		yield {
			'targets': np.array([targets], dtype=np.int32),
			'decode_length': np.array(decode_length, dtype=np.int32)
		}


