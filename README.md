# VLAx
Personal code for VLA experiments in JAX. 

The repository contains code to train with flow matching a VLA action expert attending to a VLM gemma 3 backbone. The gemma backbone has been modified for a better fit on a consumer GPU (quantization, non batching of images processing, less token per img), see : [Repository link](https://github.com/Reytuag/gemma) for more infos on the modifications.

The repository also allows to test the trained network on Libero with visualization or not. 

It also include preliminary attempts for on-policy RL to finetune the network on tasks through Flow policy optimization ([paper](https://arxiv.org/abs/2602.02481)). 

Next steps include: 
* Exploring VLA architecture and parameters changes
* Stabilizing the RL training 
* Exploring memory in VLA. (Potentially for in-context learning/meta-learning for error recovery).

![Demo](vis_videos/task1.gif)

![Attention_demo](vis_videos/attention_vis.png)


#### TOPReward tests

Added [TOPReward](https://topreward.github.io/webpage/) reward estimation test to explore this direction. Both with single image estimation (see [video](vis_videos/task9_top_reward_attempt.mp4)) as well as video (image stacking) estimation (see [video](vis_videos/task9_top_reward_video_attempt.mp4)). The results are interesting but not as clean as reported in the paper potentially due to smaller model (gemma3 4b vs Qwen3-VL-8B) + quantization + the reduction of number of token per image I use due to GPU VRAM constraints I have. 
