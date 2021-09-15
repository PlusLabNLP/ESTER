task="span_extraction"
lrs=(1e-5)
batch=(2)
seeds=(5 7 23)
device="1"
pws=(5)
model="roberta-large"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
	      for pw in "${pws[@]}"
	      do
          python code/eval_span_pred.py \
          --data_dir "./data/" \
          --model ${model} \
          --task_name  ${task} \
          --file_suffix "_ans_gen.json" \
          --device_num ${device} \
          --max_seq_length 343 \
          --fp16 \
          --learning_rate ${l} \
          --seed ${seed} \
          --model_dir "./output/${model}_batch_${s}_lr_${l}_seed_${seed}_pw_${pw}"
        done
      done
    done
done