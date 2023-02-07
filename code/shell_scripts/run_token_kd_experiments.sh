export CUDA_VISIBLE_DEVICES=1

python3 ./kd/run_experiments.py \
--experiment_type token-class-kd \
--datasets_dir /home/jcanete/new-kd/datasets \
--storage_file ./grid_token_experiments_albert_base.csv \
--student_model_name_or_path CenIA/albert-base-spanish \
--device cuda:0 \
--kd_alpha 0 \
--kd_temperature 1 \
--kd_loss_type kldiv \
--output_dir /data/jcanete/grid-kd-experiments \