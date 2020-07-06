#!/bin/sh

# Personalized model (SmallMusicVAE, LCMusicVAE)
echo "Choose your personalized model config:
    Press 1 for ae-cat-mel_2bar_big config
    Press 2 for lc-cat-mel_2bar_big config (Experimentation)
"
read -p "Waiting for input: " PVAE_CONFIG
if [[ "$PVAE_CONFIG" != "1" && "$PVAE_CONFIG" != "2" ]]; then
    echo "Option $PVAE_CONFIG is not supported"
    exit 1
fi

if [[ "$PVAE_CONFIG" == "1" ]]; then
    PVAE_CONFIG="ae-cat-mel_2bar_big"
else
    PVAE_CONFIG="lc-cat-mel_2bar_big"
fi

PVAE_TRAIN_ENTRY_POINT=$1
PVAE_GEN_ENTRY_POINT=$2
PVAE_RUN_PATH=$3
PVAE_INPUT_PATH=$4
PVAE_TRAINING_STEPS=$5
VAE_MODEL_PATH=$6
NUM_MELODY_SAMPLES=$7
MELODY_OUTPUT_PATH=$8
TRANSFORMER_UNCONDITIONED_CHECKPOINT=$9
TRANSFORMER_CONDITIONED_CHECKPOINT=${10}
TRANSFORMER_OUTPUT=${11}
TRANSFORMER_UNCONDITIONED_ENTRYPOINT=${12}
TRANSFORMER_CONDITIONED_ENTRYPOINT=${13}

# For now only supports cat-mel_2bar_big
VAE_CONFIG="cat-mel_2bar_big"
DECODER_LENGTH=1024

# Personalize Music VAE part
echo "Begin training $PVAE_CONFIG model"
python "$PVAE_TRAIN_ENTRY_POINT" --config="$PVAE_CONFIG" --run_dir="$PVAE_RUN_PATH" --mode=train --examples_path="$PVAE_INPUT_PATH" --pretrained_path="$VAE_MODEL_PATH" -num_steps="$PVAE_TRAINING_STEPS"

# Check if last character is '/'
if [[ "${PVAE_RUN_PATH: -1}" == '/' ]]; then
   PVAE_CHECKPOINTS="$PVAE_RUN_PATH""train"
else
   PVAE_CHECKPOINTS="$PVAE_RUN_PATH/train"
fi

for checkpoint in "$PVAE_CHECKPOINTS"/*; do
    echo "$checkpoint"
done

read -p "Choose your personalized MusicVAE model checkpoint from above for melody generation (Eg: model.ckpt-100): " CHOSEN_CHECKPOINT
if [[ 0 -eq $(ls "$CHOSEN_CHECKPOINT"* 2>/dev/null | wc -w) ]]; then
    echo "$CHOSEN_CHECKPOINT does not exist"
    exit 1
fi

echo "Begin generating $NUM_MELODY_SAMPLES melody samples"
python "$PVAE_GEN_ENTRY_POINT" --vae_config="$VAE_CONFIG" --config="$PVAE_CONFIG" --checkpoint_file="$CHOSEN_CHECKPOINT" --vae_checkpoint_file="$VAE_MODEL_PATH" --num_outputs="$NUM_MELODY_SAMPLES" --output_dir="$MELODY_OUTPUT_PATH"

echo "Finish generating $NUM_MELODY_SAMPLES samples"

# Music Transformer part
# Check if last character is '/'
if [[ "${MELODY_OUTPUT_PATH: -1}" == '/' ]]; then
   # Remove last character
   MELODY_SAMPLES="${MELODY_OUTPUT_PATH: 0:${#MELODY_OUTPUT_PATH}-1}"
else
   MELODY_SAMPLES="$MELODY_OUTPUT_PATH"
fi

echo "Melody sample is $MELODY_SAMPLES"

for sample in "$MELODY_SAMPLES"/*; do
    echo "$sample"
done

read -p "Choose one melody sample from above for Music Transformer input: " MELODY_SAMPLE

echo "Choose your method of music generation:
    Press 1 for sample -> primer transformer
    Press 2 for sample -> conditioned transformer
"
read -p "Waiting for input: " METHOD

if [[ "$METHOD" == "1" ]]; then
    echo "Using $MELODY_SAMPLE as primer for unconditioned music transformer"
    python "$TRANSFORMER_UNCONDITIONED_ENTRYPOINT" -model_path="$TRANSFORMER_UNCONDITIONED_CHECKPOINT" -output="$TRANSFORMER_OUTPUT" -decode_length="$DECODER_LENGTH" -primer_path="$MELODY_SAMPLE"
elif [[ "$METHOD" == "2" ]]; then
    echo "Use $MELODY_SAMPLE as conditioned melody for music transformer"
    python "$TRANSFORMER_CONDITIONED_ENTRYPOINT" -model_path="$TRANSFORMER_CONDITIONED_CHECKPOINT" -output="$TRANSFORMER_OUTPUT" -decode_length="$DECODER_LENGTH" -melody_path="$MELODY_SAMPLE"
else
    echo "Method $METHOD does not support"
    exit 1
fi
