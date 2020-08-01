# Copyright 2019 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# Modification copyright 2020 Bui Quoc Bao
# Change notebook script into package

"""Conditioned Transformer."""
import os
import time

import tensorflow.compat.v1 as tf  # pylint: disable=import-error

from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import decoding

import utils

flags = tf.flags
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
    'output_dir', None,
    'Midi output directory.'
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
flags.DEFINE_integer(
    'num_samples', 1,
    'Number of generated samples.'
)


def generate(estimator, conditional_encoders, inputs):
    """
    Generate conditioned music samples from estimator.
    :param estimator: Transformer estimator.
    :param conditional_encoders: A dictionary contains key and its encoder.
    :param inputs: Input for Transformer.
    :return:
    """
    tf.gfile.MakeDirs(FLAGS.output_dir)
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    base_name = os.path.join(
        FLAGS.output_dir,
        '%s_%s-*-of-%03d.mid' % ('melody', date_and_time, FLAGS.num_samples)
    )
    utils.LOGGER.info('Generating %d samples with format %s' % (FLAGS.num_samples, base_name))
    for i in range(FLAGS.num_samples):
        utils.LOGGER.info('Generating sample %d' % i)
        # Start the Estimator, loading from the specified checkpoint.
        input_fn = decoding.make_input_fn_from_generator(utils.melody_input_generator(
            inputs, FLAGS.decode_length))
        melody_conditioned_samples = estimator.predict(
            input_fn, checkpoint_path=FLAGS.model_path)

        # Generate sample events.
        utils.LOGGER.info('Generating sample.')
        sample_ids = next(melody_conditioned_samples)['outputs']

        # Decode to NoteSequence.
        utils.LOGGER.info('Decoding sample id')
        midi_filename = utils.decode(
            sample_ids,
            encoder=conditional_encoders['targets'])
        accompaniment_ns = utils.mm.midi_file_to_note_sequence(midi_filename)
        utils.mm.sequence_proto_to_midi_file(accompaniment_ns, base_name.replace('*', '%03d' % i))


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

    if FLAGS.output_dir is None:
        raise ValueError(
            'Required Midi output directory.'
        )

    if FLAGS.decode_length <= 0:
        raise ValueError(
            'Decode length must be > 0.'
        )

    if FLAGS.melody_path is None:
        raise ValueError(
            'Required melody Midi file path.'
        )

    problem = utils.MelodyToPianoPerformanceProblem()
    melody_conditioned_encoders = problem.get_feature_encoders()
    melody_ns = utils.get_melody_ns(FLAGS.melody_path)
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
    utils.LOGGER.info('Loading model.')
    run_config = trainer_lib.create_run_config(hparams)
    estimator = trainer_lib.create_estimator(
        FLAGS.model_name, hparams, run_config,
        decode_hparams=decode_hparams
    )

    generate(estimator, melody_conditioned_encoders, inputs)


def main(unused_argv):
    """Invoke run function, set log level."""
    utils.LOGGER.set_verbosity(FLAGS.log)
    run()


def console_entry_point():
    """Call main function."""
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
