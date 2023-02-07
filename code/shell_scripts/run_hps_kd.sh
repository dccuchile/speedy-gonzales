export CUDA_VISIBLE_DEVICES=0

python3 ./kd/run_hp_search.py \
--train_file ./datasets/PAWS-X/es/translated_train.json \
--validation_file ./datasets/PAWS-X/es/dev_2k.json \
--test_file ./datasets/PAWS-X/es/test_2k.json \
--study_name "kds-teacher-albert-base-student-albert-tiny-2" \
--experiment_type kd \
--teacher_model_name_or_path CenIA/albert-base-spanish-finetuned-pawsx \
--student_model_name_or_path CenIA/albert-tiny-spanish \
--device cuda:0 \
--num_trials 500 \