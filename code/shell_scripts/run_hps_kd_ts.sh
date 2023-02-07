export CUDA_VISIBLE_DEVICES=0

python3 ./kd/run_experiments.py \
--train_file ./datasets/PAWS-X/es/translated_train.json \
--validation_file ./datasets/PAWS-X/es/dev_2k.json \
--test_file ./datasets/PAWS-X/es/test_2k.json \
--storage_file ./gs_teachers_student_albert_base_6.csv \
--dataset_name pawsx \
--ts_student_model_name_or_path josecannete/albert-base-spanish-6 \
--device cuda:0 \
--kd_alpha 0 \
--kd_temperature 1 \
