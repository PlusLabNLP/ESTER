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

sample_size=500 #1000,2000,3000,4000

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
        python eval_ans_gen.py \
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
        --model_dir "./output/${model}_batch_${s}_lr_${l}_seed_${seed}_${sample_size}/"
        done
    done
done
