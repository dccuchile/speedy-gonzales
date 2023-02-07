export CUDA_VISIBLE_DEVICES=0

python3 kd/train.py \
--train_with_multiple_teacher_kd \
--train_file ./datasets/PAWS-X/es/translated_train.json \
--validation_file ./datasets/PAWS-X/es/dev_2k.json \
--test_file ./datasets/PAWS-X/es/test_2k.json \
--device cuda:0 \
--multiple_teacher_model_name_or_path CenIA/albert-xlarge-spanish-finetuned-pawsx CenIA/albert-xxlarge-spanish-finetuned-pawsx CenIA/bert-base-spanish-wwm-cased-finetuned-pawsx \
--student_model_name_or_path CenIA/albert-base-spanish \
--output_dir output_checkpoints/output \
--kd_alpha 0.5 \
--kd_temperature 1 \
--kd_loss_type kldiv \
--per_device_train_batch_size 64 \
--num_train_epochs 30 \
--learning_rate 5e-5 \