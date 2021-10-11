task="answer-generation"
lrs=(5e-5 1e-4 2e-4)
batch=(2 4)
seeds=(5 7 23)

device="0"
#model="google/pegasus-large"
#model="facebook/bart-base"
#model="facebook/bart-large"
#model="t5-base"
#model="allenai/unifiedqa-t5-base"
model="allenai/unifiedqa-t5-large"

#suffix="_500original"
#suffix="_500completed"
suffix=""

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
        python code/run_ans_generation_model.py \
        --data_dir "./data/" \
        --model ${model} \
        --save_model \
        --task_name  ${task} \
        --file_suffix "${suffix}.json" \
        --device_num ${device} \
        --train_batch_size ${s} \
        --num_train_epochs 10 \
        --max_seq_length 339 \
        --do_train \
        --do_eval \
        --learning_rate ${l} \
        --seed ${seed} \
        --output_dir ./output/${model}_batch_${s}_lr_${l}_seed_${seed}${suffix}
      done
    done
done