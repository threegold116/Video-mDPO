import json
import os

origin_json_path="/data/scir/sxjiang/project/Video-mDPO/data/ann_video_hal/video_llava_hound_17k.json"
new_json_path="/data/scir/sxjiang/project/Video-mDPO/data/video_llava_hound_17k_acl_gpu.json"

video_dir="/data/scir/sxjiang/dataset/hall/video"

with open(origin_json_path,"r") as f:
    data=json.load(f)
new_data=[]
for line in data:

    if os.path.exists(os.path.join(video_dir,line["video"]+".mp4")):
        line["video_path"]=os.path.join(video_dir,line["video"]+".mp4")
        new_data.append(line)
    else:
        line["video_path"]=None
        continue
print(len(data))
print(len(new_data))

with open(new_json_path,"w") as f:
    json.dump(new_data,f)