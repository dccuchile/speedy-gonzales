import json

file_in = "xquad.es.json"
file_out = "xquad-test.json"

with open(file_in, encoding="utf-8") as f_i:
    with open(file_out, "w", encoding="utf-8") as f_o:
        data = json.load(f_i)
        json.dump(data, f_o, ensure_ascii=False)

