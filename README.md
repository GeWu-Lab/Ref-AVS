# Ref-AVS
The official repo for "Ref-AVS: Refer and Segment Objects in Audio-Visual Scenes", ECCV 2024

### [Project Page](https://gewu-lab.github.io/Ref-AVS/)
### [Dataset Download](https://gewu-lab.github.io/Ref-AVS/#downloads)



### >>> Introduction
In this paper, we propose a pixel-level segmentation task called **Ref**erring **A**udio-**V**isual **S**egmentation (Ref-AVS), which requires the network to densely predict whether each pixel corresponds to the given multimodal-cue expression, including dynamic audio-visual information.

- Top-left of Fig.1 highlights the distinctions between Ref-AVS and previous tasks. 
![Fig.1 Teaser](https://github.com/GeWu-Lab/Ref-AVS/blob/main/assets/fig1.png)

- Fig.2 shows the proposed baseline model to process multimodal-cues.
![Fig.2 Baseline](https://github.com/GeWu-Lab/Ref-AVS/blob/main/assets/fig2.png)

- Fig.3 shows the statistics of this dataset.
![Fig.3 Statistics](https://github.com/GeWu-Lab/Ref-AVS/blob/main/assets/fig3.png)

### >>> Run
Run the training & evaluation:
```python
cd Ref_AVS
sh run.sh  # you should change your path configs. See /configs/config.py for more details.
```
You can download the [checkpoint](https://pan.baidu.com/s/1NrNv1hTIqI7QAvNSwl7dvw?pwd=hh58) here.

Core dependencies:
```
transformers=4.30.2
towhee=1.1.3
towhee-models=1.1.3  # Towhee is used for extracting VGGish audio feature.
```

### Citation
If you find this work useful, please consider citing it:
```
@article{wang2024refavs,
          title={Ref-AVS: Refer and Segment Objects in Audio-Visual Scenes},
          author={Wang, Yaoting and Sun, Peiwen and Zhou, Dongzhan and Li, Guangyao and Zhang, Honggang and Hu, Di},
          journal={IEEE European Conference on Computer Vision (ECCV)},
          year={2024},
}

@inproceedings{wang2024prompting,
  title={Prompting segmentation with sound is generalizable audio-visual source localizer},
  author={Wang, Yaoting and Liu, Weisong and Li, Guangyao and Ding, Jian and Hu, Di and Li, Xi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5669--5677},
  year={2024}
}
```
