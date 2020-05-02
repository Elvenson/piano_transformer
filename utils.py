# Copyright 2019 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# Modification copyright 2020 Bui Quoc Bao
# Change notebook script into package

import numpy as np
import tensorflow as tf

from tensor2tensor.data_generators import text_encoder


import magenta.music as mm
from magenta.models.score2perf import score2perf

logging = tf.compat.v1.logging


class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
	@property
	def add_eos_symbol(self):
		return True


class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
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


def melody_input_generator(inputs, decode_length):
	"""Estimator input function for melody Transformer."""
	while True:
		yield {
			'inputs': np.array([[inputs]], dtype=np.int32),
			'targets': np.zeros([1, 0], dtype=np.int32),
			'decode_length': np.array(decode_length, dtype=np.int32)
		}


def get_primer_ns(filename, max_length):
	"""
	Convert Midi file to note sequences for priming.
	:param filename: Midi file name.
	:param max_length: Maximum note sequence length for priming in seconds.
	:return:
		Note sequences for priming.
	"""
	primer_ns = mm.midi_file_to_note_sequence(filename)

	# Handle sustain pedal in primer.
	primer_ns = mm.apply_sustain_control_changes(primer_ns)

	# Trim to desired number of seconds.
	if primer_ns.total_time > max_length:
		logging.warn('Primer is longer than %d seconds, truncating.')
		primer_ns = mm.extract_subsequence(
			primer_ns, 0, max_length
		)

	# Remove drums from primer if present.
	if any(note.is_drum for note in primer_ns.notes):
		logging.warn('Primer contains drums; they will be removed.')
		notes = [note for note in primer_ns.notes if not note.is_drum]
		del primer_ns.notes[:]
		primer_ns.notes.extend(notes)

	# Set primer instrument and program.
	for note in primer_ns.notes:
		note.instrument = 1
		note.program = 0

	return primer_ns


def get_melody_ns(filename):
	"""
	Convert melody Midi file to note sequence.
	:param filename: Midi file name.
	:return:
		Melody note sequences.
	"""
	melody_ns = mm.midi_file_to_note_sequence(filename)
	melody_instrument = mm.infer_melody_for_sequence(melody_ns)
	notes = [note for note in melody_ns.notes if note.instrument == melody_instrument]
	del melody_ns.notes[:]
	melody_ns.notes.extend(
		sorted(notes, key=lambda note: note.start_time)
	)
	for i in range(len(melody_ns.notes) - 1):
		melody_ns.notes[i].end_time = melody_ns.notes[i+1].start_time

	return melody_ns


