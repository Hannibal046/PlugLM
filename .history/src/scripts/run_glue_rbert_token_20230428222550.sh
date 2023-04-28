WKDIR=

RESULTDIR=$WKDIR/results/
PT=ckpt_8000.pt
MPI="-m torch.distributed.launch --nproc_per_node 4 --master_port 12345 --use_env"


# for TASK in sst2
## different metrics for different task
echo $TASK
metrics=accuracy
dev=dev.jsonl
is_regression=""
if [ $TASK = "mrpc" ]; then
    metrics="binary_f1 accuracy"
elif [ $TASK = "cola" ]; then
    metrics=matthews_corrcoef 
elif [ $TASK = "stsb" ]; then
    metrics="pearson spearman"
    is_regression="--is_regression"
fi

## mnli matched and mismatched
if [ $TASK = "mnli_matched" ]; then
    TASK=mnli
    dev=dev_matched.jsonl
elif [ $TASK = "mnli_mismatched" ]; then
    TASK=mnli
    dev=dev_mismatched.jsonl
fi

# for MODEL in retrieval_bert_6_0_5_multi_head retrieval_bert_6_64_5_multi_head retrieval_bert_6_256_5_multi_head
# for MODEL in  retrieval_bert_6_768_5_multi_head
for MODEL in  retrieval_bert_1_0_5_multi_head
do
    CKPTDIR=$RESULTDIR/$MODEL
    for LR in 1e-5 2e-5 3e-5
    do
        for BS in 2 4 8 16 32
        do
            python $MPI \
                $WKDIR/src/run_classification.py \
                --train_file $WKDIR/data/$TASK/train.jsonl \
                --dev_file $WKDIR/data/$TASK/$dev \
                --test_file $WKDIR/data/$TASK/$dev \
                --max_seq_len 128 \
                --per_device_train_batch_size $BS \
                --learning_rate $LR \
                --num_train_epochs 10 \
                --output_dir /tmp/ \
                --logging_steps -1 \
                --metrics $metrics \
                --fix_keys \
                --fix_values \
                --values_device gpu \
                --retrieval_bert \
                --ckpt_path $CKPTDIR/ckpt/$PT \
                --config_path $CKPTDIR/config.json \
                --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
                --keys_path $CKPTDIR/keys.npy \
                --values_path $CKPTDIR/values.npy \
                $is_regression
        done
    done
done

# WKDIR=/mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert
# CKPTDIR=$WKDIR/results/retrieval_bert_6_0_5_multi_head
# PT=ckpt_7000.pt
# MPI="-m torch.distributed.launch --nproc_per_node 4 --master_port 12345 --use_env"

# TASK=cola
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics matthews_corrcoef accuracy \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=sst2
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics accuracy \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=mrpc
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics binary_f1 accuracy  \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=stsb
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --is_regression \
#     --metrics pearson spearman  \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=qqp
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics accuracy binary_f1  \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=mnli
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev_matched.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics accuracy \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=mnli
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev_mismatched.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics accuracy \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=qnli
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics accuracy  \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=rte
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics accuracy binary_f1 \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \

# TASK=wnli
# python $MPI \
#     $WKDIR/src/run_classification.py \
#     --train_file $WKDIR/data/$TASK/train.jsonl \
#     --dev_file $WKDIR/data/$TASK/dev.jsonl \
#     --max_seq_len 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/ \
#     --logging_steps -1 \
#     --metrics accuracy binary_f1 \
#     --fix_keys \
#     --fix_values \
#     --values_device gpu \
#     --retrieval_bert \
#     --ckpt_path $CKPTDIR/ckpt/$PT \
#     --config_path $CKPTDIR/config.json \
#     --vocab_path /mnt/wfs/mmchongqingwfssz/user_mqxincheng/mybert/src/vocab/vocab \
#     --keys_path $CKPTDIR/keys.npy \
#     --values_path $CKPTDIR/values.npy \
