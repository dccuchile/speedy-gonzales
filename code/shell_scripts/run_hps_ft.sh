export CUDA_VISIBLE_DEVICES=0

python3 ./kd/run_hp_search.py \
--train_file ./datasets/XNLI/train.json \
--validation_file ./datasets/XNLI/validation.json \
--test_file ./datasets/XNLI/test.json \
--study_name "test-test-test" \
--experiment_type ft \
--model_type bilstm \
--device cuda:0 \
--num_trials 500 \
--allow_pretrained_embeddings True \