task="answer-generation"
device="0"

###### Model Options ######
#model="facebook/bart-base"
#model="facebook/bart-large"
#model="t5-base"
#model="allenai/unifiedqa-t5-base"
model="allenai/unifiedqa-t5-large"


###### Additional Model Suffix ######
#suffix="_500original"
#suffix="_500completed"
suffix=""

lrs=(5e-5)
batch=(4)
seeds=(5)
root="./output"
for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
        python code/leaderboard.py \
        --data_dir "./data/" \
        --model ${model} \
        --task_name  ${task} \
        --file_suffix "_ans_gen.json" \
        --device_num ${device} \
        --eval_batch_size 8 \
        --num_train_epochs 10 \
        --max_seq_length 339 \
        --learning_rate ${l} \
        --seed ${seed} \
        --model_dir "${root}/${model}_batch_${s}_lr_${l}_seed_${seed}${suffix}/"
        done
    done
done
