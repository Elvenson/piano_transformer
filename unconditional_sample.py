# Copyright 2019 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# Modification copyright 2020 Bui Quoc Bao
# Change notebook script into package

from utils import *

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
	'log', 'INFO',
	'The threshold for what messages will be logged: '
	'DEBUG, INFO, WARN, ERROR, or FATAL.'
)
flags.DEFINE_string(
	'model_name', 'transformer',
	'The pre-trained model for sampling'
)
flags.DEFINE_string(
	'hparams_set', 'transformer_tpu',
	'Set of hparams to use'
)
flags.DEFINE_string(
	'model_path', None,
	'Pre-trained model path'
)
flags.DEFINE_string(
	'output', None,
	'Midi output path'
)
flags.DEFINE_string(
	'sample', 'random',
	'Sampling method'
)
flags.DEFINE_integer(
	'layers', 16,
	'Number of hidden layers'
)
flags.DEFINE_integer(
	'beam_size', 1,
	'Beam size for inference'
)
flags.DEFINE_integer(
	'decode_length', 1024,
	'Length of decode result'
)
flags.DEFINE_float(
	'alpha', 0.0,
	'Alpha'
)


def run():
	"""
	Load Transformer model according to flags and start sampling.
	:raises:
		ValueError: if required flags are missing or invalid.
	"""
	logging.info('Loading model')
	if FLAGS.model_path is None:
		raise ValueError(
			'Required Transformer pre-trained model path.'
		)
	if FLAGS.output is None:
		raise ValueError(
			'Required Midi output path.'
		)
	if FLAGS.decode_length <= 0:
		raise ValueError(
			'Decode length must be > 0.'
		)
	problem = PianoPerformanceLanguageModelProblem()
	unconditional_encoders = problem.get_feature_encoders()

	# Set up HParams.
	hparams = trainer_lib.create_hparams(hparams_set=FLAGS.hparams_set)
	trainer_lib.add_problem_hparams(hparams, problem)
	hparams.num_hidden_layers = FLAGS.layers
	hparams.sampling_method = FLAGS.sample

	# Set up decoding HParams.
	decode_hparams = decoding.decode_hparams()
	decode_hparams.alpha = FLAGS.alpha
	decode_hparams.beam_size = FLAGS.beam_size

	# Create Estimator.
	run_config = trainer_lib.create_run_config(hparams)
	estimator = trainer_lib.create_estimator(
		FLAGS.model_name, hparams, run_config,
		decode_hparams=decode_hparams
	)
	# Start the Estimator, loading from the specified checkpoint.
	input_fn = decoding.make_input_fn_from_generator(unconditional_input_generator([], FLAGS.decode_length))
	unconditional_samples = estimator.predict(
		input_fn, checkpoint_path=FLAGS.model_path)

	# Generate sample events.
	sample_ids = next(unconditional_samples)['outputs']

	# Decode to NoteSequence
	midi_filename = decode(
		sample_ids,
		encoder=unconditional_encoders['targets']
	)
	unconditional_ns = mm.midi_file_to_note_sequence(midi_filename)

	mm.sequence_proto_to_midi_file(unconditional_ns, FLAGS.output)


def main(unused_argv):
	logging.set_verbosity(FLAGS.log)
	run()


def console_entry_point():
	tf.compat.v1.app.run(main)


if __name__ == '__main__':
	console_entry_point()


