task="answer-generation-zero-shot"
device="0"
model="allenai/unifiedqa-t5-large"
#model="allenai/unifiedqa-t5-base"
#model="t5-base"

python code/eval_ans_gen.py \
--data_dir "./data/" \
--model ${model} \
--task_name  ${task} \
--file_suffix "_ans_gen.json" \
--device_num ${device} \
--eval_batch_size 8 \
--num_train_epochs 10 \
--max_seq_length 339

