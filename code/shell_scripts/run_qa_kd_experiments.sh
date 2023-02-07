export CUDA_VISIBLE_DEVICES=0

python3 ./kd/run_experiments.py \
--experiment_type qa-kd \
--datasets_dir /home/jcanete/new-kd/datasets \
--storage_file ./grid_qa_experiments_albert_base_10_tar.csv \
--student_model_name_or_path josecannete/albert-base-spanish-10 \
--device cuda:0 \
--kd_alpha 0 \
--kd_temperature 1 \
--kd_loss_type kldiv \
--output_dir /home/jcanete/data/grid-kd-experiments \