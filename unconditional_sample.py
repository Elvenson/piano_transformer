# Copyright 2019 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# Modification copyright 2020 Bui Quoc Bao
# Change notebook script into package

"""Unconditioned Transformer."""
from magenta.music.protobuf import music_pb2

from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

import utils

flags = utils.tf.compat.v1.flags
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
    'primer_path', None,
    'Midi file path for priming if not provided '
    'model will generate sample without priming.'
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
    'max_primer_second', 20,
    'Maximum number of time in seconds for priming.'
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

    problem = utils.PianoPerformanceLanguageModelProblem()
    unconditional_encoders = problem.get_feature_encoders()
    primer_ns = music_pb2.NoteSequence()
    if FLAGS.primer_path is None:
        targets = []
    else:
        if FLAGS.max_primer_second <= 0:
            raise ValueError(
                'Max primer second must be > 0.'
            )

        primer_ns = utils.get_primer_ns(FLAGS.primer_path, FLAGS.max_primer_second)
        targets = unconditional_encoders['targets'].encode_note_sequence(primer_ns)

        # Remove the end token from the encoded primer.
        targets = targets[:-1]
        if len(targets) >= FLAGS.decode_length:
            raise ValueError(
                'Primer has more or equal events than maximum sequence length:'
                ' %d >= %d; Aborting' % (len(targets), FLAGS.decode_length)
                )
    decode_length = FLAGS.decode_length - len(targets)

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
    utils.LOGGER.info('Loading model')
    run_config = trainer_lib.create_run_config(hparams)
    estimator = trainer_lib.create_estimator(
        FLAGS.model_name, hparams, run_config,
        decode_hparams=decode_hparams
    )

    # Start the Estimator, loading from the specified checkpoint.
    input_fn = decoding.make_input_fn_from_generator(
        utils.unconditional_input_generator(targets, decode_length))
    unconditional_samples = estimator.predict(
        input_fn, checkpoint_path=FLAGS.model_path)

    # Generate sample events.
    utils.LOGGER.info('Generating sample.')
    sample_ids = next(unconditional_samples)['outputs']

    # Decode to NoteSequence
    utils.LOGGER.info('Decoding sample id')
    midi_filename = utils.decode(
        sample_ids,
        encoder=unconditional_encoders['targets']
    )
    unconditional_ns = utils.mm.midi_file_to_note_sequence(midi_filename)

    # Append continuation to primer if any.
    continuation_ns = utils.mm.concatenate_sequences([primer_ns, unconditional_ns])

    utils.LOGGER.info('Converting note sequences to Midi file.')
    utils.mm.sequence_proto_to_midi_file(continuation_ns, FLAGS.output)


def main(unused_argv):
    """Invoke run function, set log level."""
    utils.LOGGER.set_verbosity(FLAGS.log)
    run()


def console_entry_point():
    """Call main function."""
    utils.tf.compat.v1.app.run(main)


if __name__ == '__main__':
    console_entry_point()
