import json
import re

json_data = "./train/_annotations.coco.json"
cat_id_data = "./cat_anns_coco.json"

def clean_text(text):
    return re.sub(r"[^0-9a-zA-Z]","", text)

with open(json_data, 'r', encoding="utf-8") as json_file:
    json_object = json.load(json_file)
for cat in json_object.get("categories", []):
    if "name" in cat:
        cat["name"] = clean_text(cat["name"])

with open(json_data, "w", encoding="utf-8") as json_file:
    json.dump(json_object, json_file, ensure_ascii=False, indent=2)


with open(cat_id_data, 'r', encoding="utf-8") as cat_json_file:
    cat_json_object = json.load(cat_json_file)
for cat in cat_json_object.get("categories", []):
    if "name" in cat:
        cat["name"] = clean_text(cat["name"])
with open(cat_id_data, "w", encoding="utf-8") as cat_json_file:
    json.dump(cat_json_object, cat_json_file, ensure_ascii=False, indent=2)


