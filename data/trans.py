#把原始json中的image路径给取代
import json
import os

origin_json_path="/share/home/jfliang/Datasets/mDPO-preference-data/vlfeedback_llava_10k.json"
new_json_path="/share/home/jfliang/Project/Hall/Video-mDPO/data/vlfeedback_llava_10k.json"
image_dir="/share/home/jfliang/Datasets/VLFeedback/merged_images"

with open(origin_json_path,"r") as f:
    data=json.load(f)

for line in data:
    line["img_path"]=os.path.join(image_dir,line["img_path"].split("/")[-1])
with open(new_json_path,"w") as f:
    json.dump(data,f)