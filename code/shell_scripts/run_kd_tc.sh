export CUDA_VISIBLE_DEVICES=1

python3 ./kd/train_tc.py \
--train_with_kd \
--dataset_name conll2002 \
--dataset_config_name es \
--label_column_name ner_tags \
--task_name ner \
--device cuda:0 \
--teacher_model_name_or_path CenIA/albert-xxlarge-spanish-finetuned-ner \
--student_model_name_or_path CenIA/albert-tiny-spanish \
--output_dir output_checkpoints/output \
--per_device_train_batch_size 64 \
--num_train_epochs 30 \
--learning_rate 5e-5 \
--kd_loss_type ce \