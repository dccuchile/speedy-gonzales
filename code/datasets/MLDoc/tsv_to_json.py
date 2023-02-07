import csv
import json

reader_file = "spanish.train.10000"
writer_file = "spanish.train.10000.json"

with open(reader_file, newline="") as tsvfile:
    with open(writer_file, "w") as json_file:
        reader = csv.DictReader(tsvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            row["sentence1"] = " ".join(row["sentence1"].split())
            json.dump(row, json_file, ensure_ascii=False)
            json_file.write("\n")
