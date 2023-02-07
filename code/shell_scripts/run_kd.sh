export CUDA_VISIBLE_DEVICES=1

python3 kd/train.py \
--train_with_kd \
--train_file ./datasets/PAWS-X/es/pawsx-train.json \
--validation_file ./datasets/PAWS-X/es/pawsx-dev.json \
--test_file ./datasets/PAWS-X/es/pawsx-test.json \
--device cuda:0 \
--teacher_model_name_or_path CenIA/albert-xxlarge-spanish-finetuned-pawsx \
--student_model_name_or_path CenIA/albert-tiny-spanish \
--output_dir output_checkpoints/albert-tiny-spanish-distil-x2-pawsx \
--kd_alpha 0 \
--kd_temperature 1 \
--kd_loss_type mse \
--per_device_train_batch_size 64 \
--num_train_epochs 50 \
--learning_rate 1e-4 \