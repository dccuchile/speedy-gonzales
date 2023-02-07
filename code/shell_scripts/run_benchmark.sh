export CUDA_VISIBLE_DEVICES=0,1

python3 ./kd/benchmark.py --benchmark_previous True --output_file ./benchmark_gpus_100_test.csv