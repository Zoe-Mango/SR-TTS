# RLAIF-SPA

**RLAIF-SPA** is a TTS framework that adds reinforcement learning from AI feedback to improve both **emotional expressiveness** and **semantic accuracy**.It rewards prosodic–emotional alignment (structure, emotion, speed, tone) and clear speech, outperforming Chat-TTS on LibriSpeech with lower WER and higher human ratings.

## Abstract

Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the  **RLAIF-SPA** framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages **Prosodic Label Alignment** to enhance expressive quality by jointly considering semantic accuracy and prosodic–emotional alignment along four fine-grained dimensions: **Structure, Emotion, Speed**, and **Tone**. In addition, it incorporates {**Semantic Accuracy Feedback** to ensure the generation of clear and accurate speech. Our framework diagram is as follows.

![image-20250919000152146](https://github.com/Zoe-Mango/RLAIF-SPA/blob/main/assets/overview.png)

## Installation

```bash
conda env create -f RLAIF-SPA-environment.yml
```

## data

We have provided the data that will be used, stored in the **data** folder.

## main

You can also use the training script we provide to perform the training.
