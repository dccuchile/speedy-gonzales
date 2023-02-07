# Shell Scripts

These directory contains multiple shell scripts used during the project:

- run_benchmark.sh -> used to measure MACs, number of parameters, inference time
- run_ft_qa.sh -> can be used to fine-tune a model on question answering
- run_ft_tc.sh -> can be used to fine-tune a model on token classification tasks (ner, pos)
- run_ft.sh -> can be used to fine-tune a model on text classification tasks (mldoc, pawsx, xnli)
- run_hps_ft.sh -> 
- run_hps_kd_ts.sh ->
- run_hps_kd.sh ->
- run_kd_qa.sh -> can be used to apply knowledge distillation on question answering
- run_kd_tc.sh -> can be used to apply knowledge distillation on token classification tasks (ner, pos)
- run_kd.sh -> can be used to apply knowledge distillation on text classification tasks (mldoc, pawsx, xnli)
- run_mt_kd.sh -> can be used to apply knowledge distillation on text classification tasks (mldoc, pawsx, xnli) with multiple teachers (not sure is completely working though)
- run_qa_kd_experiments.sh -> used to run multiple experiments (a grid search) to obtain the best results of knowledge distillation on question answering datasets
- run_seq_kd_experiments.sh -> used to run multiple experiments (a grid search) to obtain the best results of knowledge distillation on text classification tasks (mldoc, pawsx, xnli)
- run_token_kd_experiments.sh -> used to run multiple experiments (a grid search) to obtain the best results of knowledge distillation on token classification tasks (ner, pos)
