# from collections.abc import Callable
import json
from pathlib import Path
import random
import re
import csv
# from typing import Any, Iterator, Optional
from typing import Iterator, Optional

import wandb
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    GenerationConfig,
    WavLMModel
)
from MiniCPM_o.modeling_minicpmo import MiniCPMO
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

# from accelerate import Accelerator
import torchaudio
import jiwer

from transformers import AutoProcessor, AutoModelForAudioClassification
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# 初始化 Qwen Audio
def load_qwen_audio(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    model.eval()
    return processor, model

# 分析合成语音风格
def analyze_audio_tags(audio_path: str, processor, model, device='cuda'):
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = logits.argmax(dim=-1)

    # 假设模型是 multi-label 多任务分类模型
    # 返回的是4个标签：[结构, 情感, 语速, 语调]
    return predicted_ids[0].tolist()  # [3, 1, 2, 0] for example

def load_whisper(model_name="openai/whisper-base", device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model

def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    bf16: bool = True,
    device_map=None,
    use_lora: bool = False,
    lora_config: Optional[LoraConfig] = None,
) -> tuple[MiniCPMO, PreTrainedTokenizer]:
    from peft import get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    model = MiniCPMO.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation='sdpa',
        torch_dtype=torch.float16
        # torch_dtype=torch.float32

    )
    if use_lora:
        assert lora_config is not None, "LoRA config required"
        model = get_peft_model(model, lora_config)
    return model, tokenizer


def get_wavlm_model(model_path):
    return WavLMModel.from_pretrained(model_path)

import uuid
import torchaudio
import os

def generate_and_extract_embedding(
    model, tokenizer, instruction: str, wavlm_model, tmp_dir="/tmp"
):
    """
    使用 MiniCPM 当前模型生成语音（基于 instruction），保存到临时 .wav 文件，
    然后用 WavLM 提取语音 embedding。
    """
    # 创建唯一输出路径
    tmp_wav_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")

    # 构造 chat 输入
    msgs = [{'role': 'user', 'content': [instruction]}]

    # 使用当前模型生成语音
    model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        max_new_tokens=128,
        use_tts_template=True,
        generate_audio=True,
        temperature=0.3,
        output_audio_path=tmp_wav_path,
    )

    # 读取语音并提取 WavLM embedding
    waveform, sr = torchaudio.load(tmp_wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    waveform = waveform.to(wavlm_model.device)
    with torch.no_grad():
        embedding = wavlm_model(input_values=waveform, output_hidden_states=True).hidden_states[-1]

    # 清理临时文件
    os.remove(tmp_wav_path)

    return torch.squeeze(embedding)



@torch.no_grad()
def rollout_single(
    model: MiniCPMO,
    wavlm_model: WavLMModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    id: str,
    structure: str,
    emotion: str,
    speech_speed: str,
    tone: str,
    qwen_processor,
    qwen_model,
    model_device,
    whisper_processor,
    whisper_model,
    num_rollouts: int,
    device: str,
    # max_length: int = 256,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()
    model = model.to(device)
    audio_path = id  # oracle_answer 是传进来的 id
    id = os.path.splitext(os.path.basename(audio_path))[0]  # 提取出纯 ID

    # task是文本，oracle_answer是id

    # 1. format prompt
    # chat_messages = [
    #     {
    #         "role": "system",
    #         "content": system_prompt,
    #     },
    #     {
    #         "role": "user",
    #         "content": task,
    #     },
    # ]
    # print(f"看一下chat_mesages:{chat_messages}")
    chat_messages = [{'role':'user', 'content':[text]}]
    # chat_prompt = tokenizer.apply_chat_template(
    #     chat_messages, tokenize=False, add_generation_prompt=True
    # )
    # print(f"看一下task：{task}")
    model_inputs = model._prepare_inputs(
        msgs=chat_messages,
        tokenizer=tokenizer,
        sampling=True,
        use_tts_template=True,
        generate_audio=True,
        temperature=0.3,
        # max_inp_length=512
        max_inp_length=256

    )

    for key_, value_ in model_inputs.items():
        if type(value_) == torch.Tensor:
            model_inputs[key_] = model_inputs[key_].repeat(num_rollouts, 1)
        elif type(value_) == list:
            if len(value_) != 0:
                res = []
                for i in range(num_rollouts):
                    res.append(value_[0])
                model_inputs[key_] = res

    input_ids = model_inputs["input_ids"]

    if "inputs_embeds" in model_inputs:
        embeds = model_inputs["inputs_embeds"]
        if not torch.isfinite(embeds).all():
            print("[ERROR] inputs_embeds 包含 NaN 或 Inf，跳过该样本")
            return None, None, None, None



    # 2. sample completions
    # print(f"看一下tokenizer：{tokenizer.decode([0])}")
    pad_token_id = tokenizer.eos_token_id
    # print(f"看一下pad_token_id：{pad_token_id}")
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    result, outputs = model.generate(
            **model_inputs,
            generation_config=generation_config,
            tokenizer=tokenizer
        )
    sequence_ids = outputs.sequences
    # completions = tokenizer.decode(
    #     sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=False
    # )
    # print(f"看一下completions:{completions}")
    # print(f"看一下sequence_ids：{sequence_ids}")
    # print(f"看一下input_ids.shape:{input_ids.shape}")
    # print(f"tokenizer.decode(result):{tokenizer.decode(sequence_ids[0])}")
    
    action_mask = torch.ones_like(sequence_ids, dtype=torch.bool)
    # action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask[sequence_ids == 0] = False
    action_mask = action_mask[:, 1:]
    # print(f"看一下action_mask：{action_mask}")

    # def text_to_audio_to_text(model_inputs, outputs, results, text_to_audio_model):
    #     """
    #         拿到的是模型生成的文本转化成为语音向量准备进行simo的计算
    #         生成文本-mel-音频(waveform)-提取相似度
    #         将生成文本变为音频waveform(再用于 WavLM 向量提取)
    #     """

    #     mel_spec = text_to_audio_model._generate_mel_spec(model_inputs, outputs, results)
    #     wav_numpy, sr = text_to_audio_model.get_wav_numpy_and_sr(mel_spec)
    #     if sr != 16000:
    #         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    #         waveform = resampler(wav_numpy)
        
    #     return torch.squeeze(waveform)
    
    
    def audio_to_feature(id, origin_parent, audio_to_feature_model):
        """
        将原始的语音文件转为 WavLM 向量表示(参考embedding)
        """
        # audio_name = id + ".wav"
        audio_name = id if id.endswith(".wav") else id + ".wav"

        audio_path = os.path.join(origin_parent, audio_name)
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform.to(audio_to_feature_model.device)

        embedding = audio_to_feature_model(input_values=waveform, output_hidden_states=True).hidden_states[-1]
        return torch.squeeze(embedding)


    def compute_cos(origin_audio_vectors: list[torch.Tensor], gen_audio_vectors: list[torch.Tensor]) -> torch.Tensor:

        assert len(origin_audio_vectors) == len(gen_audio_vectors), "origin 和 gen 数量必须一致"

        cos_scores = torch.zeros(len(origin_audio_vectors), dtype=torch.float)

        for i, (ori_emb, gen_emb) in enumerate(zip(origin_audio_vectors, gen_audio_vectors)):
            ori_emb = torch.squeeze(ori_emb)
            gen_emb = torch.squeeze(gen_emb)
            min_len = min(ori_emb.shape[0], gen_emb.shape[0])
            emb1 = ori_emb[:min_len, :]
            emb2 = gen_emb[:min_len, :]
            cos = F.cosine_similarity(emb1, emb2, dim=-1).mean()
            cos_scores[i] = cos.item()

        return cos_scores
    _simple_number_map = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90
}
    def simple_text2num(phrase):
        """仅支持20~99之间由两个词组合的数字"""
        words = phrase.lower().strip().split()
        total = 0
        for word in words:
            if word in _simple_number_map:
                total += _simple_number_map[word]
        return str(total) if total > 0 else phrase

    def normalize_numbers(text):
        text = text.lower()
        pattern = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|' \
                r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|' \
                r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|' \
                r'eighty|ninety)(?:\s(?:one|two|three|four|five|six|seven|eight|nine))?\b'

        return re.sub(pattern, lambda m: simple_text2num(m.group()), text)
    def get_transformation():
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            normalize_numbers,
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip()
        ])
        return transformation
    
    def transcribe_with_whisper(audio_path: str, processor, model, device="cuda"):
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    
    def compute_wer(id: str, origin_text_path: str, gen_text_path: str, transformation) -> torch.Tensor:
        ori_text = None
        gen_text = None

        with open(origin_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                content = json.loads(line)
                if content.get("id") == id:
                    ori_text = content.get("text")
                    break

        with open(gen_text_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["id"] == id:
                    gen_text = row["gen_text"]
                    break

        if ori_text is None:
            raise ValueError(f"[compute_wer] 未找到参考文本 id={id}")
        if gen_text is None:
            raise ValueError(f"[compute_wer] 未找到识别文本 id={id}")
        
        # 标准化文本
        ori_clean = transformation(ori_text)
        gen_clean = transformation(gen_text)

        print(f"[DEBUG][WER] 当前样本 id: {id}")
        print(f"[DEBUG][WER] 找到 origin_text: {ori_text}")
        print(f"[DEBUG][WER] 找到 gen_text: {gen_text}")
        print(f"[DEBUG][WER] 找到 ori_clean: {ori_clean}")
        print(f"[DEBUG][WER] 找到 gen_clean: {gen_clean}")
        assert ori_text is not None, f"未找到原始文本 for id={id}"
        assert gen_text is not None, f"未找到生成文本 for id={id}"

        # 计算 WER
        wer_score = jiwer.wer(ori_clean, gen_clean)

        return torch.tensor([wer_score], dtype=torch.float)
       
    def compute_tag_match_reward(gen_audio_path: str, target_labels: dict, processor, qwen_model, device) -> float:
        pred_labels = analyze_audio_tags(gen_audio_path, processor, qwen_model, device=device)
        label_keys = ["structure", "emotion", "speech_speed", "tone"]
        ground_truth = [
            target_labels["structure"],
            target_labels["emotion"],
            target_labels["speech_speed"],
            target_labels["tone"]
        ]

        # 映射分类值（你需要自己定义）
        label_mapping = {
            "structure": {"simple": 0, "complex": 1},
            "emotion": {"happy": 0, "sad": 1, "angry": 2, "neutral": 3},
            "speech_speed": {"fast": 0, "medium": 1, "slow": 2},
            "tone": {"serious": 0, "casual": 1, "humorous": 2}
        }

        matches = 0
        for i, key in enumerate(label_keys):
            gt = label_mapping[key].get(ground_truth[i].lower())
            if gt is not None and gt == pred_labels[i]:
                matches += 1
        return matches / len(label_keys)  # 平均一致度得分

    def compute_reward(cos_res, wer_scores, tag_rewards=None):
        simo_mapped = cos_res + 1.0  # 将 SIMO 从 [-1, 1] 映射到 [0, 2]      
        wer_complement = 1.0 - wer_scores # WER 补值为 (1 - wer)，即正确率   

        rewards = 0.4 * simo_mapped + 0.2 * wer_complement

        if tag_rewards is not None:
            tag_rewards_tensor = torch.tensor(tag_rewards, dtype=torch.float, device=cos_res.device)
            rewards += 0.4 * tag_rewards_tensor

        rewards = torch.clamp(rewards, min=0.0, max=2.0)
        print(f"final rewards: {rewards}")
        return rewards




    # gen_feature_embedding = text_to_audio_to_text(model_inputs, outputs, results, model)
    origin_path = "train-clean-100-1k-wav_files"
    gen_path = "output path"
    # origin_feature_embedding = audio_to_feature(id, origin_path, wavlm_model)
    # gen_feature_embedding = audio_to_feature(id, gen_path, wavlm_model)
    origin_embeddings = []
    gen_embeddings = []
    for i in range(num_rollouts):
        origin_embeddings.append(audio_to_feature(id, origin_path, wavlm_model))
        gen_embeddings.append(audio_to_feature(id, gen_path, wavlm_model))


    # cos_results = compute_cos([origin_feature_embedding], [gen_feature_embedding])
    cos_results = compute_cos(origin_embeddings, gen_embeddings)


    transformation = get_transformation()
    origin_text_path = "librispeech_train-clean-100_texts1k.json"
    # wers_ = compute_wer(id, origin_text_path, gen_text_path, transformation)
    # wers_list = []
    # for i in range(num_rollouts):
    #     wers_list.append(compute_wer(id, origin_text_path, gen_text_path, transformation))
    wers_list = []

    output_path = "your own path"

    for i in range(num_rollouts):
        gen_audio_path = os.path.join(output_path, f"{id}.wav")
        gen_text = transcribe_with_whisper(gen_audio_path, whisper_processor, whisper_model, device=device)

        ori_text = None
        with open(origin_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                content = json.loads(line)
                if content.get("id") == id:
                    ori_text = content.get("text")
                    break

        if ori_text is None:
            raise ValueError(f"[compute_wer] 未找到参考文本 id={id}")

        ori_clean = transformation(ori_text)
        gen_clean = transformation(gen_text)
        wer = jiwer.wer(ori_clean, gen_clean)
        wers_list.append(torch.tensor([wer], dtype=torch.float))

    wers_ = torch.stack(wers_list).view(-1)

    tag_rewards = []
    for i in range(num_rollouts):
        gen_audio = os.path.join(output_path, f"{id}.wav")
        reward = compute_tag_match_reward(
            gen_audio_path=gen_audio,
            target_labels={
                "structure": structure,
                "emotion": emotion,
                "speech_speed": speech_speed,
                "tone": tone,
            },
            processor=qwen_processor,
            qwen_model=qwen_model,
            device=model_device
        )
        tag_rewards.append(reward)

    
    returns = compute_reward(cos_results, wers_, tag_rewards).view(-1)
    # 加入归一化（z-score 标准化）
    # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    # returns = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)


    wandb.log({
        "reward_mean": returns.mean().item(),
        "reward_max": returns.max().item(),
        "reward_min": returns.min().item(),
        "WER_mean": wers_.mean().item(),
        "SIMO_mean": cos_results.mean().item()
    })

    print(f"WER scores: {wers_}")
    print(f"SIMO scores: {cos_results}")
    print(f"Final rewards: {returns}")


    # print(f"看一下奖励：{returns}")
    # return sequence_ids, returns.to(sequence_ids.device), action_mask, result
    return sequence_ids, returns, action_mask, result

def rollout(
    model: MiniCPMO,
    wavlm_model: WavLMModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    ids: list[str],
    structures: list[str],
    emotions: list[str],
    speech_speeds: list[str],
    tones: list[str],
    qwen_processor,
    qwen_model,
    num_rollouts: int,
    device: str,
    max_length: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    all_sequence_ids = []
    all_returns = []
    all_action_masks = []
    all_completions = []

    for text, id, structure, emotion, speech_speed, tone in zip(
    texts, ids, structures, emotions, speech_speeds, tones
):
        try:
            result = rollout_single(
                model=model,
                wavlm_model=wavlm_model,
                tokenizer=tokenizer,
                text=text,
                id=id,
                structure=structure,
                emotion=emotion,
                speech_speed=speech_speed,
                tone=tone,
                qwen_processor=qwen_processor,
                qwen_model=qwen_model,
                num_rollouts=num_rollouts,
                device=device,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )
            if result[0] is not None:
                all_sequence_ids.append(result[0])
                all_returns.append(result[1])
                all_action_masks.append(result[2])
                all_completions.extend(result[3])
        except Exception as e:
            print(f"[rollout] 跳过样本 {id}，出错: {e}")
            continue

    return (
        torch.cat(all_sequence_ids, dim=0),
        torch.cat(all_returns, dim=0),
        torch.cat(all_action_masks, dim=0),
        all_completions
    )

def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: MiniCPMO,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    device = attention_mask.device
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.to(device)
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)


    output = model.llm.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )

    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(csv_path: str, parent_path="/CPM/output/label/"):
    rows = []
    import pandas as pd
    from datasets import Dataset
    df = pd.read_csv(csv_path)
 # 提取各类标签
    df["structure"] = df["labels"].str.extract(r"1\.\s*Structure:\s*(.*?)\s*2\.")[0]
    df["emotion"] = df["labels"].str.extract(r"2\.\s*Emotion:\s*(.*?)\s*3\.")[0]
    df["speech_speed"] = df["labels"].str.extract(r"3\.\s*Speech Speed:\s*(.*?)\s*4\.")[0]
    df["tone"] = df["labels"].str.extract(r"4\.\s*Tone:\s*(.*)")[0]

    def build_instruction(row):
        try:
            instruction = f"Please express the sentence '{row['text']}' with a structure that is {row['structure']}, " \
                          f"emotionally {row['emotion']}, at a speech speed of {row['speech_speed']}, and in a {row['tone']} tone, " \
                          f"voiced by a middle-aged woman."
            return instruction
        except:
            return row["text"]  # fallback to plain text

    df["prompt"] = df.apply(build_instruction, axis=1)

    rows = [{
        "prompt": row["prompt"],
        "id": os.path.join(parent_path, row["id"] if row["id"].endswith(".wav") else row["id"] + ".wav"),
        "structure": row["structure"],
        "emotion": row["emotion"],
        "speech_speed": row["speech_speed"],
        "tone": row["tone"]
    } for _, row in df.iterrows()]

    return rows





def main():
    # accelerator = Accelerator()
    # device = accelerator.device
    model_device = torch.device("cuda:4")
    ref_device = torch.device("cuda:5")
    whisper_device = torch.device("cuda:2")
    qwen_device = torch.device("cuda:3") 

    num_epochs = 10  # 训练10轮

    seed = 42
    wandb_project = None  # "tiny_grpo"
    device_index = 0
    model_name = "MiniCPM-o"
    qwen_model_path =  "Qwen2-Audio" 
    qwen_processor, qwen_model = load_qwen_audio(qwen_model_path)
    qwen_model.to(qwen_device)
    whisper_processor, whisper_model = load_whisper("whisper-large-v3", device=model_device)

    # LoRA 配置和目标模块
    from peft import LoraConfig, TaskType
    lora_target_modules = []
    for i in range(28):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        lora_target_modules.append(f"llm.model.layers.{i}.self_attn.{proj}")
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            lora_target_modules.append(f"llm.model.layers.{i}.mlp.{proj}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # checkpoint_path = Path("./output")
    checkpoint_path = Path("/CPM/checkpoint")

    checkpoint_interval = 20
    train_batch_size = 1
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    # group_size = 2
    group_size = 4
    rollouts_per_step = 1
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    # max_length = 512
    max_length = 256
    top_p = 1.0
    temperature = 1.0

    # device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # Accelerate prepare
    # accelerator = Accelerator()
    wavlm_model_path = "wavlm"
    wavlm_model = get_wavlm_model(wavlm_model_path).to(wavlm_device)

    reference_model, _ = load_model(model_name)
    reference_model.to(ref_device) 
    reference_model.eval()  # 确保不参与反向传播
    # reference_model.init_tts()
    # reference_model.tts.float()
    # model, tokenizer = load_model(model_name)
    model, tokenizer = load_model(
    model_name,
    use_lora=True,
    lora_config=lora_config
    )
    model.to(model_device)
    model.print_trainable_parameters()
  
    model.init_tts()
    model.tts.float()
    #看一下层
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print(name)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id
    csv_path = "librispeech_train-clean-100_texts1k.csv"
    prompts = read_prompts(csv_path)
    # print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    # 将需要放在多卡上的变量使用accelerator包裹
    # prompt_loader, model, reference_model, optimizer = accelerator.prepare(prompt_loader, reference_model, optimizer)
    global_step = 0  # 全局步数
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch + 1}/{num_epochs} ===")

        for k, prompt_batch in enumerate(prompt_loader):
            rollout_returns = []

            replay_buffer.clear()

            batch = prompt_batch[0]
            texts = [batch["prompt"]]
            ids = [batch["id"]]
            structures = [batch["structure"]]
            emotions = [batch["emotion"]]
            speech_speeds = [batch["speech_speed"]]
            tones = [batch["tone"]]


            with torch.no_grad():
                    sequence_ids, returns, action_mask, completions = rollout(
                        model.to(model_device),
                        wavlm_model.to(wavlm_device),
                        tokenizer,
                        texts=texts,
                        ids=ids,
                        structures=structures,
                        emotions=emotions,
                        speech_speeds=speech_speeds,
                        tones=tones,
                        qwen_processor=qwen_processor,
                        qwen_model=qwen_model,
                        whisper_processor=whisper_processor,
                        whisper_model=whisper_model,
                        num_rollouts=group_size,
                        device=model_device,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                    )


                    sequence_ids = sequence_ids.to(model_device)
                    returns = returns.to(model_device)
                    action_mask = action_mask.to(model_device)


                    # print(
                    #     f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                    # )
                    rollout_returns.append(returns.cpu())
                    # print(f"看一下returns：{returns}")
                    # advantages = group_advantages(returns).to(model_device)
                    advantages = group_advantages(returns).view(-1).to(model_device)
                    if torch.all(advantages == 0):
                        print("Skipping batch: all advantages are 0.")
                        continue
                    returns = returns.view(-1) 
                    # attention_mask = sequence_ids != torch.tensor(pad_token_id, device=sequence_ids.device)
                    attention_mask = sequence_ids != torch.full_like(sequence_ids, pad_token_id, device=sequence_ids.device)

                    log_probs = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids.to(model_device),
                        attention_mask=attention_mask.to(model_device),
                    )
                    log_probs_ref = sequences_log_probs(
                        model=reference_model,
                        sequence_ids=sequence_ids.to(ref_device),
                        attention_mask=attention_mask.to(ref_device),
                    )
                    kl = approx_kl_divergence(
                        log_probs=log_probs,
                        log_probs_ref=log_probs_ref.to(log_probs.device),
                        action_mask=action_mask,
                    )

                    experience = Experience(
                        sequences=sequence_ids,
                        action_log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        returns=returns,
                        advantages=advantages,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        kl=kl,
                    )
                    # replay_buffer.append(experience.to(cpu_device))
                    replay_buffer.append(experience)

            torch.cuda.empty_cache()
            episode_return_sum = torch.stack(rollout_returns).sum()
            print(f"returns of step {k}: {episode_return_sum:.4f}")
            wandb.log({"returns": episode_return_sum})

            experience_sampler = DataLoader(
                replay_buffer,
                batch_size=train_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=join_experience_batch,
            )

            for step_epoch in range(epochs_per_step):
                model.train()
                total_loss = 0.0
                total_steps = 0

                for exp in experience_sampler:
                    exp: Experience

                    # exp = exp.to(accelerator.device)  # 标记
                    exp = exp.to(next(model.parameters()).device)
                    optimizer.zero_grad()

                    log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                    )

                    loss, kl = objective(log_probs=log_probs, experience=exp)
                    print(f"loss以及kl：{loss},{kl}")
                    if not loss.isfinite() or not kl.isfinite():
                        print(f"Loss not finite, skipping backward, loss={loss}, kl={kl}")
                        print(f"experience.advantages={experience.advantages}")
                        continue

                    # loss.backward()
                    # accelerator.backward(loss)
                    loss.backward()

                    total_loss += loss.item()
                    total_steps += 1
                    
                    grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    print(f"epoch{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                    wandb.log({"kl": kl, "grad_norm": grad_norm})

                    optimizer.step()
                    
                if total_steps > 0:
                    avg_loss = total_loss / total_steps
                    print(f"Epoch {epoch+1}, step {k}, average loss: {avg_loss:.6f}")
                    wandb.log({"avg_loss": avg_loss})

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            save_dir = checkpoint_path / f"step_{global_step}"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[Checkpoint] Saved model at step {global_step} to {save_dir}")
            
        global_step += 1
            
        print(f"[Epoch {epoch+1}] avg_loss: {avg_loss:.6f}, avg_return: {episode_return_sum.item() / len(prompt_loader):.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "epoch_avg_loss": avg_loss,
            "epoch_avg_return": episode_return_sum.item() / len(prompt_loader),
        })

        
    # 保存最终模型
    if checkpoint_path is not None:
        final_dir = checkpoint_path / "final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"[INFO] Final model saved to: {final_dir}")



if __name__ == "__main__":
    main()
