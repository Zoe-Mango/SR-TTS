# SR-TTS

Self-Reward TTS (SR-TTS) is an emotional text-to-speech framework that lets an LLM act as its own judge.It optimizes both semantic accuracy and prosodic-emotional alignment (structure, emotion, speed, tone), boosting expressiveness and intelligibility.【如果挂到arxiv连接放到这里】

## Introduction

Text-to-speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the SR-TTS framework, which incorporates a Self-Reward mechanism to employ LLM itself to judge semantic accuracy and prosodic-emotional label alignment as the reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Emotion-Driven Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic–emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Intelligibility-Driven Feedback via Semantic Accuracy to ensure the generation of clear and accurate speech.

## Installation

```bash
conda env create -f qwen_audio_environment.yml
```

## data

We have provided the data that will be used, stored in the **data** folder.

## main

You can also use the training script we provide to perform the training.
