# Project / Paper Introduction
This is the project repo for our EMNLP'21 paper: 

Here, we provide brief descriptions of the final data and detailed instructions to reproduce results in our paper. For more details, please refer to the paper.

# Data
Final data used for the experiments are saved in `./data/` folder with train/dev/test splits. Most data fields are straightforward. Just a few notes,
- **question_event**: this field is not provided by annotators nor used for our experiments. We simply use some heuristic rules based on POS tags to extract possible events in the questions. Users are encourages to try alternative tools such semantic role labeling. 
- **original_events** and **indices** are the annotator-provided event triggers plus their indices in the context.
- **answer_texts** and **answer_indices** (in train and dev) are the annotator-provided answers plus their indices in the context.

#### Please Note the evaluation script below (II) only works for the dev set for now as we are currently setting up a leaderboard for evaluating the test set.
#### Instruction for submissions will appear here shortly. Thanks for your patience.

# Models
## I. Install packages. 
We list the packages in our environment in env.yml file for your reference. Below are a few key packages.
- python=3.8.5
- pytorch=1.6.0
- transformers=3.1.0
- cudatoolkit=10.1.243
- apex=0.1

## II. Replicate results in our paper.
### 1. Download trained models.
For reproduction purpose, we release all trained models.
- Download link: https://drive.google.com/drive/folders/1bTCb4gBUCaNrw2chleD4RD9JP1_DOWjj?usp=sharing. 
- We only provide models with the best "hyper-parameters", and each comes with three random seeds: 5, 7, 23.
- Make several directories to save models `./output/`, `./output/facebook/` and `./output/allenai/`.
- For BART models, download them into `./output/facebook/`.
- For UnifiedQA models, download them into `./output/allenai/`.
- All other models can be saved in `./output/` directly. These ensure evaluation scripts run properly below.

### 2. Zero-shot performances in Table 3. 
Run `bash ./code/eval_zero_shot.sh`. Model options are provided in the script.

### 3. Generative QA Fine-tuning performances in Table 3.
Run `bash ./code/eval_ans_gen.sh`. Make sure the following arguments are set correctly in the script.
- Model Options provided in the script
- Set suffix=""
- Set `lrs` and `batch` according to model options. You can find these numbers in Appendix G of the paper.

### 4. Figure 6: UnifiedQA-large model trained with sub-samples.
Run bash ./code/eval_ans_gen.sh`. Make sure the following arguments are set correctly in the script.
- `model="allenai/unifiedqa-t5-large"`
- `suffix={"_500" | "_1000" | "_2000" | "_3000" | "_4000"}`
- Set `lrs` and `batch` accordingly. You can find these information in the folder name containing the trained model objects.

### 5. Table 4: 500 original annotations v.s. completed
- `bash ./code/eval_ans_gen.sh` with `model="allenai/unifiedqa-t5-large` and `suffix="_500original`
- `bash ./code/eval_ans_gen.sh` with `model="allenai/unifiedqa-t5-large` and `suffix="_500completed`
- Set `lrs` and `batch` accordingly again.

### 6. Extractive QA Fine-tuning performances in Table 3.
Simply run `bash ./code/eval_span_pred.sh` as it is.

### 7. Figure 8: Extractive QA Fine-tuning performances by changing positive weights.
- Run `bash ./code/eval_span_pred.sh`.
- Set `pw`, `lrs` and `batch` according to model folder names again.


## III. Model Training
We also provide the model training scripts below.

### 1. Generative QA: Fine-tuning in Table 3.
- Run `bash ./code/run_ans_generation.sh`. 
- Model options and hyper-parameter search range are provided in the script.
- We use `--fp16` argument to activate apex for GPU memory efficient training except for UnifiedQA-t5-large (trained on A100 GPU).

### 2. Figure 6: UnifiedQA-large model trained with sub-samples.
- Run `bash ./code/run_ans_gen_subsample.sh`.
- Set `sample_size` variable accordingly in the script.

### 3. Table 4: 500 original annotations v.s. completed
- Run `bash ./code/run_ans_gen.sh` with `model="allenai/unifiedqa-t5-large` and `suffix="_500original`
- Run `bash ./code/run_ans_gen.sh` with `model="allenai/unifiedqa-t5-large` and `suffix="_500completed`

### 4. Extractive QA Fine-tuning in Table 3 + Figure 8
Simply run `bash ./code/run_span_pred.sh` as it is.



