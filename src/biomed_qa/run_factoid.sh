WKDIR=
MODEL=${1}

export SAVE_DIR=$WKDIR/results/bioqa/$MODEL
export DATA_DIR=$WKDIR/data/bio/QA/BioASQ
export OFFICIAL_DIR=$WKDIR/src/biomed_qa/scripts/bioasq_eval

export BATCH_SIZE=12
export LEARNING_RATE=8e-6
export NUM_EPOCHS=3
export MAX_LENGTH=384
export SEED=0
export CUDA_VISIBLE_DEVICES=0

# Train
python run_factoid.py \
    --model_type bert \
    --model_name_or_path $WKDIR/results/$MODEL \
    --do_train \
    --train_file ${DATA_DIR}/BioASQ-train-factoid-7b.json \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_seq_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --output_dir ${SAVE_DIR} \
    --do_lower_case

# Evaluation
python run_factoid.py \
    --model_type bert \
    --model_name_or_path ${SAVE_DIR} \
    --do_eval \
    --predict_file ${DATA_DIR}/BioASQ-test-factoid-7b.json \
    --golden_file ${DATA_DIR}/7B_golden.json \
    --per_gpu_eval_batch_size ${BATCH_SIZE} \
    --max_seq_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --official_eval_dir ${OFFICIAL_DIR} \
    --output_dir ${SAVE_DIR} \
    --do_lower_case