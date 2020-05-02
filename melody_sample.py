from utils import *

from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
	'log', 'INFO',
	'The threshold for what messages will be logged: '
	'DEBUG, INFO, WARN, ERROR, or FATAL.'
)
flags.DEFINE_string(
	'model_name', 'transformer',
	'The pre-trained model for sampling.'
)
flags.DEFINE_string(
	'hparams_set', 'transformer_tpu',
	'Set of hparams to use.'
)
flags.DEFINE_string(
	'model_path', None,
	'Pre-trained model path.'
)
flags.DEFINE_string(
	'melody_path', None,
	'Midi file path for melody'
)
flags.DEFINE_string(
	'output', None,
	'Midi output path.'
)
flags.DEFINE_string(
	'sample', 'random',
	'Sampling method.'
)
flags.DEFINE_integer(
	'layers', 16,
	'Number of hidden layers.'
)
flags.DEFINE_integer(
	'beam_size', 1,
	'Beam size for inference.'
)
flags.DEFINE_integer(
	'decode_length', 1024,
	'Length of decode result.'
)
flags.DEFINE_float(
	'alpha', 0.0,
	'Alpha for decoder.'
)


def run():
	"""
	Load Transformer model according to flags and start sampling.
	:raises:
		ValueError: if required flags are missing or invalid.
	"""
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

	if FLAGS.melody_path is None:
		raise ValueError(
			'Required melody Midi file path.'
		)

	problem = MelodyToPianoPerformanceProblem()
	melody_conditioned_encoders = problem.get_feature_encoders()
	melody_ns = get_melody_ns(FLAGS.melody_path)
	inputs = melody_conditioned_encoders['inputs'].encode_note_sequence(
		melody_ns)

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
	logging.info('Loading model')
	run_config = trainer_lib.create_run_config(hparams)
	estimator = trainer_lib.create_estimator(
		FLAGS.model_name, hparams, run_config,
		decode_hparams=decode_hparams
	)

	# Start the Estimator, loading from the specified checkpoint.
	input_fn = decoding.make_input_fn_from_generator(melody_input_generator(inputs, FLAGS.decode_length))
	melody_conditioned_samples = estimator.predict(
		input_fn, checkpoint_path=FLAGS.model_path)

	# Generate sample events.
	logging.info('Generating sample.')
	sample_ids = next(melody_conditioned_samples)['outputs']

	# Decode to NoteSequence.
	logging.info('Decoding sample id')
	midi_filename = decode(
		sample_ids,
		encoder=melody_conditioned_encoders['targets'])
	accompaniment_ns = mm.midi_file_to_note_sequence(midi_filename)

	logging.info('Converting note sequences to Midi file.')
	mm.sequence_proto_to_midi_file(accompaniment_ns, FLAGS.output)


def main(unused_argv):
	logging.set_verbosity(FLAGS.log)
	run()


def console_entry_point():
	tf.compat.v1.app.run(main)


if __name__ == '__main__':
	console_entry_point()
