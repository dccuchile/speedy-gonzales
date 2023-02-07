import contextlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from thop import profile
from tqdm import tqdm
from transformers import AutoConfig, HfArgumentParser, PretrainedConfig

from models.ffnn import FFNNConfig
from stats_utils import filter_outliers
from utils import get_model_from_config

logger = logging.getLogger()


class PerformanceBenchmark(object):
    def __init__(
        self,
        model_config: PretrainedConfig,
        task: str = "sequence_classification",
    ) -> None:
        self.config = model_config
        self.model = get_model_from_config(task, config=self.config)

    def compute_size(self) -> dict[str, float]:
        state_dict = self.model.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        # Delete temporary file
        tmp_path.unlink()
        # print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def parameter_count(self) -> dict[str, Any]:
        return {"model_parameters": sum(p.numel() for p in self.model.parameters())}

    def flops_count(self, query: torch.Tensor) -> dict[str, Any]:
        # The library we are using is full of prints we don't need so the following
        # context manager will disable them
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            macs, _ = profile(self.model, inputs=(query,))
        return {"macs": macs}

    def time_pipeline(
        self,
        query: torch.Tensor = None,
        batch_size: int = None,
        sequence_length: int = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        if query is None:
            if batch_size is None or sequence_length is None:
                raise ValueError(
                    "This method requires a query or (batch size and sequence length)"
                )
            else:
                query = torch.randint(
                    low=0,
                    high=self.config.vocab_size,
                    size=(batch_size, sequence_length),
                    device=device,
                )

        latencies = []
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                self.model(query)
            # Timed run
            for _ in range(100):
                start_time = perf_counter()
                self.model(query)
                latency = perf_counter() - start_time
                latencies.append(latency)

        # We convert the latencies to ms
        latencies = np.array(latencies) * 1000
        # We filter outliers of the latencies array
        filtered_latencies = filter_outliers(latencies)
        # Compute run statistics
        time_avg_ms = np.mean(filtered_latencies)
        time_std_ms = np.std(filtered_latencies)
        # print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {
            "time_avg_ms": time_avg_ms,
            "time_std_ms": time_std_ms,
            # Just in case, we are going to save the latencies for further analysis
            "latencies": latencies,
        }

    def get_queries(
        self, batch_sizes: list[int], sequence_lengths: list[int]
    ) -> list[torch.Tensor]:
        queries = []
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                queries.append(
                    torch.randint(
                        low=0,
                        high=self.config.vocab_size,
                        size=(batch_size, sequence_length),
                    )
                )
        return queries

    def run_benchmark(
        self, batch_sizes: list[int], sequence_lengths: list[int], device: str = "cpu"
    ) -> dict[str, Any]:
        self.model.to(device)
        self.model.eval()

        metrics = {}  # type: dict[str, Any]
        queries = self.get_queries(batch_sizes, sequence_lengths)

        metrics.update(self.compute_size())
        metrics.update(self.parameter_count())
        metrics["times"] = list()

        for query in tqdm(queries, miniters=1, desc="Query: "):
            query = query.to(device)
            # tqdm.write(f"Query shape: {query.shape}")
            batch_size, sequence_length = query.shape
            time_dict = {"batch_size": batch_size, "sequence_length": sequence_length}
            try:
                time_dict.update(self.time_pipeline(query=query, device=device))
                time_dict.update(self.flops_count(query=query))
            except RuntimeError:  # This means an out of memory error.
                time_dict.update(
                    {
                        "time_avg_ms": None,
                        "time_std_ms": None,
                        "macs": None,
                        "latencies": None,
                    }
                )
            metrics["times"].append(time_dict)

        return metrics


class BenchmarkComparison:
    def __init__(
        self,
        config_ids: list[str],
        configs: list[PretrainedConfig],
        batch_sizes: list[int],
        sequence_lengths: list[int],
        task: str,
        devices: list[str] = ["cpu"],
    ) -> None:
        self.config_ids = config_ids
        self.configs = configs
        self.batch_sizes = batch_sizes
        self.sequence_lengths = sequence_lengths
        self.task = task
        self.devices = devices

    def run(self) -> pd.DataFrame:
        """
        Run the benchmark for every model in the different settings and returns
        a DataFrame containing the results in each area.
        """
        comparison = []
        for config_id, config in tqdm(
            zip(self.config_ids, self.configs),
            total=len(self.config_ids),
            miniters=1,
            desc="Configs: ",
        ):
            for device in tqdm(self.devices, miniters=1, desc="Devices: "):
                # tqdm.write(f"Running on config_id: {config_id} and device: {device}")
                benchmark = PerformanceBenchmark(config, task=self.task)
                results = benchmark.run_benchmark(
                    self.batch_sizes, self.sequence_lengths, device=device
                )
                comparison.append(
                    {
                        "config_id": config_id,
                        "device": torch.cuda.get_device_properties(device).name
                        if "cuda" in device
                        else device,
                        "benchmark": results,
                    }
                )
        comparison_df = self.to_dataframe(comparison)
        return comparison_df

    def to_dataframe(self, comparison: list[dict]) -> pd.DataFrame:
        records = []
        for benchmark in comparison:
            config_id = benchmark["config_id"]
            device = benchmark["device"]
            size_mb = benchmark["benchmark"]["size_mb"]
            parameter_count = benchmark["benchmark"]["model_parameters"]
            for time_dict in benchmark["benchmark"]["times"]:
                batch_size = time_dict["batch_size"]
                sequence_length = time_dict["sequence_length"]
                time_avg_ms = time_dict["time_avg_ms"]
                time_std_ms = time_dict["time_std_ms"]
                latencies = time_dict["latencies"]
                macs = time_dict["macs"]
                # flops = time_dict["flops"]
                # activation_flops = time_dict["activation_flops"]

                records.append(
                    {
                        "config_id": config_id,
                        "device": device,
                        "size_mb": size_mb,
                        "parameter_count": parameter_count,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "time_avg_ms": time_avg_ms,
                        "time_std_ms": time_std_ms,
                        "macs": macs,
                        "latencies": latencies,
                        # "flops": flops,
                        # "activation_flops": activation_flops,
                    }
                )

        return pd.DataFrame.from_records(records)


def benchmark_previous_models(output_file) -> None:
    config_mapping = {
        "beto-uncased": "dccuchile/bert-base-spanish-wwm-uncased",
        "beto-cased": "dccuchile/bert-base-spanish-wwm-cased",
        "distilbeto": "CenIA/distillbert-base-spanish-uncased",
        "albeto-tiny": "CenIA/albert-tiny-spanish",
        "albeto-base": "CenIA/albert-base-spanish",
        "albeto-large": "CenIA/albert-large-spanish",
        "albeto-xlarge": "CenIA/albert-xlarge-spanish",
        "albeto-xxlarge": "CenIA/albert-xxlarge-spanish",
        "bertin-roberta-base": "bertin-project/bertin-roberta-base-spanish",
        "bne-roberta-base": "PlanTL-GOB-ES/roberta-base-bne",
        "bne-roberta-large": "PlanTL-GOB-ES/roberta-large-bne",
    }
    config_ids = []  # type: list[str]
    configs = []  # type: list[PretrainedConfig]
    for key, value in config_mapping.items():
        config_ids.append(key)
        configs.append(AutoConfig.from_pretrained(value, num_labels=3))

    benchmark = BenchmarkComparison(
        config_ids=config_ids,
        configs=configs,
        batch_sizes=[1, 2, 4, 8, 16, 32, 64],
        sequence_lengths=[128, 256, 512],
        task="sequence_classification",
        devices=["cuda:0", "cuda:1"],
    )
    comparison_df = benchmark.run()
    comparison_df.to_csv(output_file, index=False)


def benchmark_new_models(output_file) -> None:
    config_mapping = {
        "albeto-tiny-6": "josecannete/albert-tiny-spanish-6",
        "albeto-tiny-8": "josecannete/albert-tiny-spanish-8",
        "albeto-tiny-12": "josecannete/albert-tiny-spanish-12",
        "albeto-base-2": "josecannete/albert-base-spanish-2",
        "albeto-base-4": "josecannete/albert-base-spanish-4",
        "albeto-base-6": "josecannete/albert-base-spanish-6",
        "albeto-base-8": "josecannete/albert-base-spanish-8",
        "albeto-base-10": "josecannete/albert-base-spanish-10",
    }
    config_ids = []  # type: list[str]
    configs = []  # type: list[PretrainedConfig]
    for key, value in config_mapping.items():
        config_ids.append(key)
        configs.append(AutoConfig.from_pretrained(value, num_labels=3))

    benchmark = BenchmarkComparison(
        config_ids=config_ids,
        configs=configs,
        batch_sizes=[1],  # [1, 2, 4, 8, 16, 32, 64]
        sequence_lengths=[128, 256, 512],
        task="sequence_classification",
        devices=["cpu", "cuda:0"],  # ["cpu", "cuda:0", "cuda:1"]
    )
    comparison_df = benchmark.run()
    comparison_df.to_csv(output_file, index=False)


@dataclass
class BenchmarkArguments:
    output_file: str = field(
        metadata={"help": "The path of the output csv."},
    )
    benchmark_previous: bool = field(
        default=False, metadata={"help": "Whether to benchmark previous models or not."}
    )
    benchmark_new: bool = field(
        default=False,
        metadata={
            "help": (
                "Wheter to benchmark the new models (alberts with more or less layers)"
                " or not."
            )
        },
    )


def main():
    parser = HfArgumentParser((BenchmarkArguments,))
    (benchmark_args,) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("benchmark.log"),
        ],
    )
    logger.setLevel(logging.INFO)

    if benchmark_args.benchmark_previous:
        benchmark_previous_models(benchmark_args.output_file)
        return
    elif benchmark_args.benchmark_new:
        benchmark_new_models(benchmark_args.output_file)
        return

    config1 = FFNNConfig(vocab_size=31000, num_labels=10)
    config2 = FFNNConfig(vocab_size=100000, num_labels=10)

    benchmark = BenchmarkComparison(
        config_ids=["ffnn_vocab_31k", "ffnn_vocab_100k"],
        configs=[config1, config2],
        batch_sizes=[8, 16, 32, 64],
        sequence_lengths=[128, 512],
        task="sequence_classification",
        devices=["cpu", "cuda:0", "cuda:1"],
    )
    comparison_df = benchmark.run()
    comparison_df.to_csv(benchmark_args.output_file, index=False)


if __name__ == "__main__":
    main()
