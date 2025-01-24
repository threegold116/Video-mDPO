1. 环境安装
trl install (huggingface/trl)
0.7.2
2. base跑通
python bunny/run_mdpo_bunny.py
3. 参考LLaVA复制trl/llava
- trl和LLaVA的保持一致比较好，但不知道是哪一个版本的
- transformers Trainer的版本也和llava保持一致
- less is more, 不应该安装trl的这些包的（
transformers 4.38.2

4. debug MDPO代码
- 数据的处理代码：在MDPO的data加载基础上改
- 损失的计算代码：在MDPO的trainer基础上改
- 模型的处理代码：需要处理forward部分

5. MDPO代码流程
- 数据加载
chosen和rejected的答案
图像
输入的prompt

- 数据记载
在trainer的dataloader和mdpo的datacollator得到input_ids

- 损失计算
a. 在trainer的compute_loss中，调用mdpo内的损失计算
b. 调用mdpo的get_batch_metrics
c. 在mdpo的get_batch_metrics中，调用mdpo的concated_metrics
先进行第一次chosen+rejected文本的推理，计算文本偏好概率
然后再进行chosen的imageless图像的推理，计算图像偏好概率
然后进行reference_model的推理，计算参考概率
得到概率后，计算MDPO损失

- 梯度更新

6. 修改代码
a. 再bunny_mdpo_run.py的基础上修改得到llavaov_mdpo_run.py
b. 修改llava_qwen.py加上crop_images函数
c. 修改config.yaml
d. 修改merge_lora.py
e. vit的lora要去掉吗？


7. 代码说明
video_mdpo_trainer.py: 保留mdpo的损失，改为视频
video_mdpo_trainer_2.py: 加入扰动文本损失