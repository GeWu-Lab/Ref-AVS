# Ref-AVS
The official repo for "Ref-AVS: Refer and Segment Objects in Audio-Visual Scenes", ECCV 2024

### >>> Introduction
In this paper, we propose a pixel-level segmentation task called **Ref**erring **A**udio-**V**isual **S**egmentation (Ref-AVS), which requires the network to densely predict whether each pixel corresponds to the given multimodal-cue expression, including dynamic audio-visual information.

- Top-left of Fig.1 highlights the distinctions between Ref-AVS and previous tasks. 
![Fig.1 Teaser](https://github.com/GeWu-Lab/Ref-AVS/blob/main/assets/fig1.png)

- Fig.2 shows the proposed baseline model to process multimodal-cues.
![Fig.2 Baseline](https://github.com/GeWu-Lab/Ref-AVS/blob/main/assets/fig2.png)

### >>> Run
Run the training & evaluation:
```python
cd Ref_AVS
sh run.sh  # you should change your path configs. See /configs/config.py for more details.
```

Core dependencies:
```
transformers=4.30.2
towhee=1.1.3
towhee-models=1.1.3  # Towhee is used for extracting VGGish audio feature.
```

