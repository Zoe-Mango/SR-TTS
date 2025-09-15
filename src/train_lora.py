from collections.abc import Callable
import json
from pathlib import Path
import random
import re
import csv
from typing import Any, Iterator, Optional
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
import torch.nn.functional as F

from MiniCPM_o.modeling_minicpmo import MiniCPMO
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

# from accelerate import Accelerator
import torchaudio
import jiwer
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

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



@torch.no_grad()
def rollout_single(
    model: MiniCPMO,
    wavlm_model: WavLMModel,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    device: str,
    max_length: int = 256,
    # max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()
    model = model.to(device)
    audio_path = oracle_answer  # oracle_answer 是传进来的 id
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
    chat_messages = [{'role':'user', 'content':[task]}]
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

    # duplicate prompt num_rollouts times

    # model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
    #     num_rollouts, 1
    # )

    # input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    # model_inputs["input_ids"] = input_ids

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
    # def audio_to_text(id, origin_parent, audio_to_text_model):
    #     """
    #         将原始的语音转化成为语音向量准备进行计算
    #         读取原始音频-embedding
    #         将参考音频文件变为WavLM的向量表示(embedding)
    #     """

    #     audio_name = id + ".wav"
    #     audio_path = os.path.join(origin_parent, audio_name)
    #     waveform = model(input_values=wav1, output_hidden_states=True).hidden_states[-1]
    #     waveform = torch.squeeze(waveform)
    #     return waveform


    
    # def compute_cos(origin_audio_vector, gen_audio_vector):

    #     num_rollout_gen = len(gen_audio_path.shape)

    #     origin_length = gen_audio_vector.size(1)
    #     gen_audio_vector = gen_audio_vector.size(1)

    #     min_length = min(origin_length, gen_audio_vector)
        
    #     cos_res = torch.zeros(num_rollout_gen, 1, dtype=torch.float)
        
    #     origin_audio_vector = torch.squeeze(origin_audio_vector)
        
    #     for i, completion in enumerate(gen_audio_vector):
    #         emb1 = origin_audio_vector[:min_length,:]
    #         emb2 = i[:min_length, :]
    #         cos_ = F.cosine_similarity(emb1, emb2)[0].item()
    #         cos_res[i] = cos_
    #     return cos_res

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

    def compute_wer(id: str, origin_text_path: str, gen_text_path: str, transformation) -> torch.Tensor:
        ori_text = None
        gen_text = None

        with open(origin_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                content = json.loads(line)
                if content.get("id") == id:
                    ori_text = content.get("text")
                    break

        # with open(gen_text_path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         content = json.loads(line)
        #         if content.get("id") == id:
        #             gen_text = content.get("text")
        #             break

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
       
        # num_rollout_gen = len(gen_text_path)
        # wer_scores = torch.zeros(num_rollout_gen, 1, dtype=torch.float)
        # for i,j in enumerate(gen_text_path):
        #     hyp_clean = transformation(j)
        #     wer = jiwer.wer(ref_clean, hyp_clean)
        #     wer_scores[i] = wer
        # return wer_scores
    
    # def compute_reward(cos_res, wer_scores):
    #     rewards = (cos_res + 1) / 2
    #     rewards -= wer_scores
    #     rewards -= 0.1 * wer_scores
    #     print(f"再看一下rewards:{rewards}")
    #     return rewards
    def compute_reward(cos_res, wer_scores):
        simo_mapped = cos_res + 1.0  # 将 SIMO 从 [-1, 1] 映射到 [0, 2]      
        wer_complement = 1.0 - wer_scores # WER 补值为 (1 - wer)，即正确率   
        rewards = 0.7 * simo_mapped + 0.3 * wer_complement # 计算 reward，设权重
        print(f"cos_res (SIMO): {cos_res}")
        print(f"mapped SIMO: {simo_mapped}")
        print(f"WER: {wer_scores}")
        print(f"1 - WER: {wer_complement}")
        print(f"final rewards: {rewards}")
        rewards = torch.clamp(rewards, min=0.0, max=2.0) #加入 clamp 防止越界
        return rewards

    # 3. determine rewards test
    # returns = torch.zeros(num_rollouts, 1, dtype=torch.float, device=device)
    # for i, completion in enumerate(result):
    #     if i % 2 == 0:
    #         reward = 0.5
    #     else:
    #         reward = 1

    #     returns[i] = reward




    # gen_feature_embedding = text_to_audio_to_text(model_inputs, outputs, results, model)
    origin_path = "/train-clean-100-1k-wav_files"
    gen_path = "/CPM/output/base"
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
    gen_text_path = "/CPM/eval/output/gen_base_text.csv"
    # wers_ = compute_wer(id, origin_text_path, gen_text_path, transformation)
    wers_list = []
    for i in range(num_rollouts):
        wers_list.append(compute_wer(id, origin_text_path, gen_text_path, transformation))
    wers_ = torch.stack(wers_list).view(-1)

    
    returns = compute_reward(cos_results, wers_).view(-1)
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
    tasks: list[str],
    oracle_answers: list[str],
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

    for task, oracle_answer in zip(tasks, oracle_answers):
        try:
            result = rollout_single(
                model=model,
                wavlm_model=wavlm_model,
                tokenizer=tokenizer,
                task=task,
                oracle_answer=oracle_answer,
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
            print(f"[rollout] 跳过样本 {oracle_answer}，出错: {e}")
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

# def sequences_log_probs(
#     model: MiniCPM

# ) -> torch.Tensor:



def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# def read_prompts(
#     file_name: str,
#     predicate: Optional[Callable[[Any], bool]] = None,
#     max_rows: Optional[int] = None,
# ) -> list:
#     rows = []
#     for x in read_jsonl(file_name):
#         if predicate is None or predicate(x):
#             rows.append(x)
#         if max_rows is not None and len(rows) >= max_rows:
#             break
#     return rows

def read_prompts(csv_path: str, parent_path="/CPM/output/base/"):
    rows = []
    import pandas as pd
    from datasets import Dataset
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.rename_column("id", "id")
    dataset = dataset.rename_column("text", "prompt")
    # dataset = dataset.map(lambda x: {"id": parent_path + x["id"] + ".wav"})
    dataset = dataset.map(lambda x: {
    "id": os.path.join(parent_path, x["id"] if x["id"].endswith(".wav") else x["id"] + ".wav")
})

    for i in dataset:
        rows.append(i)
    return rows





def main():
    # accelerator = Accelerator()
    # device = accelerator.device
    model_device = torch.device("cuda:1")
    ref_device = torch.device("cuda:2")
    wavlm_device = torch.device("cuda:3")
    num_epochs = 10  # 训练10轮

    seed = 42
    wandb_project = None  # "tiny_grpo"
    device_index = 0
    model_name = "MiniCPM-o"
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

    checkpoint_path = Path("/CPM/base_cpt/checkpoint")

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

            questions = prompt_batch["prompt"]
            answers = prompt_batch["id"]

            with torch.no_grad():


                    sequence_ids, returns, action_mask, completions = rollout(
                        model.to(model_device),
                        wavlm_model.to(wavlm_device),
                        tokenizer,
                        tasks=questions,              # 现在是 list[str]
                        oracle_answers=answers,       # 现在是 list[str]
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
