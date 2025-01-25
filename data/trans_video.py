#把原始json中的video路径给取代
import json
import os

origin_json_path="/share/home/jfliang/Project/Video_hallucination/video_instruction_train_dpo_sft_dpo_17k.jsonl"
new_json_path="/share/home/jfliang/Project/Hall/Video-mDPO/data/video_llava_hound_17k.json"
perturbation_dir="/share/home/jfliang/Project/Video_hallucination/perturbation_dpo"
video_dir="/share/home/jfliang/Datasets/llava-hound/video"
with open(origin_json_path,"r") as f:
    data=f.readlines()
    data = [json.loads(line) for line in data]
    print(data[0])  
    if os.path.exists(os.path.join(video_dir,data[0]["video"]+".mp4")):
        print("video exists")
    else:
        print("video not exists")
new_data=[]
non_video_id=set()
for line in data:
    for file in os.listdir(perturbation_dir):
        if line["video"] in file:
            with open(os.path.join(perturbation_dir,file),"r") as f:
                perturbation_data=json.load(f)
                line["perturbation"]=perturbation_data["perturbation"]
                break
    if "perturbation" not in line:
        continue
    
    
    if os.path.exists(os.path.join(video_dir,line["video"]+".mp4")):
        line["video_path"]=os.path.join(video_dir,line["video"]+".mp4")
        new_data.append(line)
    else:
        non_video_id.add(line["video"])
        print(data[0]["video"])
        print("video not exists")
print(len(new_data))
print(len(data))
with open(new_json_path,"w") as f:
    json.dump(new_data,f)