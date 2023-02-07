export CUDA_VISIBLE_DEVICES=1

python3 ./kd/train_tc.py \
--dataset_name conll2002 \
--dataset_config_name es \
--label_column_name ner_tags \
--task_name ner \
--device cuda:0 \
--model_name_or_path CenIA/albert-tiny-spanish \
--output_dir output_checkpoints/output \
--per_device_train_batch_size 64 \
--num_train_epochs 10 \
--learning_rate 5e-5 \