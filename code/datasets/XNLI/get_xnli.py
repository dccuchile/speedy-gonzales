from datasets import load_dataset
import jsonlines

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}


def save_as_json(dataset, split_name):
    print(f"Saving {len(dataset[split_name])} examples in {split_name}.json")
    with jsonlines.open(f"{split_name}.json", mode="w") as writer:
        for row in dataset[split_name]:
            to_write = {
                "sentence1": row["premise"],
                "sentence2": row["hypothesis"],
                "label": row["label"],
                #"label": id2label[row["label"]],
            }
            writer.write(to_write)


def main():
    dataset = load_dataset("xnli", "es")
    for split in dataset:
        save_as_json(dataset=dataset, split_name=split)


if __name__ == "__main__":
    main()
