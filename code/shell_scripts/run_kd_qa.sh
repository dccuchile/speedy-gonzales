export CUDA_VISIBLE_DEVICES=1

python3 ./kd/train_qa.py \
--train_with_kd \
--loading_script_path /home/jcanete/new-kd/datasets/QA/qa_datasets.py \
--train_file /home/jcanete/new-kd/datasets/QA/SQAC/sqac-train.json \
--validation_file /home/jcanete/new-kd/datasets/QA/SQAC/sqac-dev.json \
--test_file /home/jcanete/new-kd/datasets/QA/SQAC/sqac-test.json \
--device cuda:0 \
--student_model_name_or_path CenIA/albert-base-spanish \
--teacher_model_name_or_path CenIA/albert-xxlarge-spanish-finetuned-qa-sqac \
--output_dir output_checkpoints/output \
--per_device_train_batch_size 64 \
--num_train_epochs 50 \
--learning_rate 1e-4 \