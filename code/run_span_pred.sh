task="span_extraction"
lrs=(1e-5)
batch=(4)
seeds=(23)
device="3"
pws=(1 2 5 10 20)

model="roberta-large"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
	      for pw in "${pws[@]}"
	      do
           nohup python code/run_span_pred.py \
            --data_dir "./data/" \
            --model ${model} \
            --task_name  ${task} \
            --file_suffix "_ans_gen.json" \
            --device_num ${device} \
            --train_batch_size ${s} \
            --num_train_epochs 10 \
            --pos_weight ${pw} \
            --max_seq_length 343 \
            --do_train \
            --do_eval \
            --fp16 \
            --learning_rate ${l} \
            --seed ${seed} \
            --output_dir "./output/${model}_batch_${s}_lr_${l}_seed_${seed}_pw_${pw}_rep" \
            > code/logs/roberta-large_batch_${s}_lr_${l}_seed_${seed}_pw_${pw}_rep
        done
      done
    done
done