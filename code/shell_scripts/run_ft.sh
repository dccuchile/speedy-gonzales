export CUDA_VISIBLE_DEVICES=0

python3 ./kd/train.py \
--train_file ./datasets/XNLI/train.json \
--validation_file ./datasets/XNLI/validation.json \
--test_file ./datasets/XNLI/test.json \
--device cuda:0 \
--model_name_or_path output_checkpoints/random_albert_tiny \
--output_dir output_checkpoints/output \
--per_device_train_batch_size 64 \
--num_train_epochs 10 \
--learning_rate 5e-5 \