import json
from pathlib import Path
import random
import re
import csv
from typing import Iterator, Optional
import subprocess
import sys
import os
import uuid
import tempfile
import numpy as np

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    GenerationConfig,
)
from MiniCPM_o.modeling_minicpmo import MiniCPMO
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

import torchaudio
import jiwer

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


# 输出路径配置
OUTPUT_BASE = "output path"
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "audio")
WER_RESULT_CSV = os.path.join(OUTPUT_BASE, "wer_result.csv")
AUDIO_LABEL_RESULT_JSON = os.path.join(OUTPUT_BASE, "audio_label_result.json")

# 创建输出目录
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# 跨环境配置
QWEN_AUDIO_ENV_PATH = "envoronment/qwen_audio/bin/python" 
QWEN_AUDIO_SCRIPT_PATH = "the path of qwen_audio_service.py"       

 
class OrderedDataIterator:
    """有序数据迭代器，支持断点续传"""
    
    def __init__(self, data_list, start_index=0):
        self.data_list = data_list
        self.current_index = start_index
        self.total_length = len(data_list)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= self.total_length:
            raise StopIteration
        
        item = self.data_list[self.current_index]
        self.current_index += 1
        return item, self.current_index - 1  # 返回数据和索引
    
    def __len__(self):
        return self.total_length - self.current_index
    
    def get_current_index(self):
        return self.current_index
    
    def set_current_index(self, index):
        self.current_index = max(0, min(index, self.total_length))
        
    def reset(self):
        self.current_index = 0
        
    def is_finished(self):
        return self.current_index >= self.total_length


class QwenAudioAnalyzer:
    """Qwen Audio 分析器 - 跨环境调用版本"""
    
    def __init__(self):
        self._service_ready = False
        self._warmup_done = False
        
    def _ensure_service_ready(self):
        """确保服务进程准备就绪 - 预热服务"""
        if not self._warmup_done:
            try:
                print("[INFO] 预热 Qwen Audio 服务...")
                # 创建一个临时的小音频文件进行测试
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # 创建1秒的静音音频
                    import numpy as np
                    import soundfile as sf
                    
                    dummy_audio = np.zeros(16000, dtype=np.float32)
                    sf.write(tmp_file.name, dummy_audio, 16000)
                    
                    # 预热调用
                    cmd = [QWEN_AUDIO_ENV_PATH, QWEN_AUDIO_SCRIPT_PATH, tmp_file.name]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    # 清理临时文件
                    os.unlink(tmp_file.name)
                    
                    if result.returncode == 0:
                        self._warmup_done = True
                        print("[INFO] Qwen Audio服务预热完成")
                    else:
                        print(f"[WARNING] Qwen Audio服务预热失败: {result.stderr}")
                        
            except Exception as e:
                print(f"[WARNING] 服务预热失败: {e}")
    
    def analyze_audio_tags(self, audio_path: str) -> dict:
        """通过subprocess调用qwen_audio环境分析音频标签"""
        try:
            # 确保服务预热
            self._ensure_service_ready()
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"[ERROR] 音频文件不存在: {audio_path}")
                return {
                    "structure": "-1",
                    "emotion": "-1", 
                    "speech_speed": "-1",
                    "tone": "-1"
                }
                
            # 检查文件大小
            if os.path.getsize(audio_path) == 0:
                print(f"[ERROR] 音频文件为空: {audio_path}")
                return {
                    "structure": "-1",
                    "emotion": "-1", 
                    "speech_speed": "-1",
                    "tone": "-1"
                }
            
            # 调用qwen_audio环境中的脚本
            cmd = [QWEN_AUDIO_ENV_PATH, QWEN_AUDIO_SCRIPT_PATH, audio_path]
            print(f"[DEBUG] 执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 增加超时时间
            
            print(f"[DEBUG] 返回码: {result.returncode}")
            
            if result.returncode == 0:
                # 解析JSON输出
                stdout_lines = result.stdout.strip().split('\n')
                json_line = None
                
                # 查找JSON行（通常是最后一行）
                for line in reversed(stdout_lines):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        json_line = line
                        break
                
                if json_line:
                    try:
                        analysis_result = json.loads(json_line)
                        print(f"[DEBUG] 成功解析JSON: {analysis_result}")
                        return analysis_result
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] JSON解析失败: {e}")
                        print(f"[DEBUG] 尝试解析的内容: {json_line}")
                else:
                    print("[ERROR] 未找到有效的JSON输出")
                    if result.stdout:
                        print(f"[DEBUG] 完整stdout: {result.stdout}")
                    
                # 如果没有找到JSON，返回默认值
                return {
                    "structure": "-1",
                    "emotion": "-1", 
                    "speech_speed": "-1",
                    "tone": "-1"
                }
            else:
                print(f"[ERROR] Qwen Audio分析失败，返回码: {result.returncode}")
                if result.stderr:
                    print(f"[ERROR] 错误信息: {result.stderr}")
                return {
                    "structure": "-1",
                    "emotion": "-1", 
                    "speech_speed": "-1",
                    "tone": "-1"
                }
                
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Qwen Audio分析超时: {audio_path}")
            return {
                "structure": "-1",
                "emotion": "-1", 
                "speech_speed": "-1",
                "tone": "-1"
            }
        except Exception as e:
            print(f"[ERROR] 音频标签分析失败 {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "structure": "-1",
                "emotion": "-1", 
                "speech_speed": "-1",
                "tone": "-1"
            }
    
    def compute_tag_match_reward(self, audio_path: str, target_labels: dict) -> float:
        """计算标签匹配奖励"""
        try:
            # 分析音频标签
            pred_labels = self.analyze_audio_tags(audio_path)
            
            # 计算匹配度
            matches = 0
            total_labels = 4
            
            # 标准化比较
            for key in ["structure", "emotion", "speech_speed", "tone"]:
                target_value = str(target_labels.get(key, "")).lower().strip()
                pred_value = str(pred_labels.get(key, "")).lower().strip()
                
                # 处理特殊情况
                if key == "speech_speed":
                    # 处理语速标签的不同表示方式
                    target_value = target_value.replace(" ", "_")
                    pred_value = pred_value.replace(" ", "_")
                
                print(f"[DEBUG] 标签比较 - {key}: target='{target_value}', predicted='{pred_value}'")
                
                # 如果预测值不是-1且与目标值匹配，则算作匹配
                if target_value and pred_value != "-1" and pred_value == target_value:
                    matches += 1
                    print(f"[DEBUG] ✓ {key} 匹配成功")
                else:
                    print(f"[DEBUG] ✗ {key} 匹配失败")
            
            reward = matches / total_labels if total_labels > 0 else 0.0
            
            # 保存分析结果
            result_data = {
                "audio_path": audio_path,
                "predicted_labels": pred_labels,
                "target_labels": target_labels,
                "matches": matches,
                "total": total_labels,
                "reward": reward,
                "timestamp": str(torch.cuda.current_stream().query()) if torch.cuda.is_available() else "0"
            }
            
            # 追加到JSON文件
            self._save_audio_label_result(result_data)
            
            print(f"[INFO] 标签匹配分析 - 音频: {os.path.basename(audio_path)}")
            print(f"[INFO] 匹配结果: {matches}/{total_labels}, 奖励: {reward:.3f}")
            print(f"[INFO] 预测标签: {pred_labels}")
            print(f"[INFO] 目标标签: {target_labels}")
            
            return reward
            
        except Exception as e:
            print(f"[ERROR] 标签匹配计算失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _save_audio_label_result(self, result_data):
        """保存音频标签分析结果"""
        try:
            # 读取现有数据
            if os.path.exists(AUDIO_LABEL_RESULT_JSON):
                with open(AUDIO_LABEL_RESULT_JSON, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []
            
            # 添加新数据
            existing_data.append(result_data)
            
            # 写回文件
            with open(AUDIO_LABEL_RESULT_JSON, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[ERROR] 保存音频标签结果失败: {e}")


class WERCalculator:
    """WER计算器"""
    
    def __init__(self):
        self.transformation = self._get_transformation()
        # 初始化CSV文件
        self._init_wer_csv()
    
    def _init_wer_csv(self):
        """初始化WER结果CSV文件"""
        if not os.path.exists(WER_RESULT_CSV):
            with open(WER_RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'ori_text', 'gen_text', 'ori_clean', 'gen_clean', 'wer'])
    
    def _get_transformation(self):
        """获取文本标准化转换器"""
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
        
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            normalize_numbers,
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip()
        ])
        return transformation
    
    def transcribe_with_whisper(self, audio_path: str, processor, model, device="cuda:4"):
        """使用Whisper转录音频"""
        try:
            print(f"[DEBUG] Whisper转录音频: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000)
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"[DEBUG] Whisper转录结果: {transcription}")
            return transcription.strip()
        except Exception as e:
            print(f"[ERROR] Whisper转录失败 {audio_path}: {e}")
            return ""
    
    def compute_wer(self, id: str, ori_text: str, gen_text: str) -> float:
        """计算WER并保存结果"""
        try:
            # 标准化文本
            ori_clean = self.transformation(ori_text)
            gen_clean = self.transformation(gen_text)
            
            # 计算WER
            wer = jiwer.wer(ori_clean, gen_clean)
            
            # 保存到CSV
            with open(WER_RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([id, ori_text, gen_text, ori_clean, gen_clean, wer])
            
            print(f"[INFO] WER计算 - ID: {id}, WER: {wer:.3f}")
            print(f"[INFO] 原文: {ori_text}")
            print(f"[INFO] 识别: {gen_text}")
            print(f"[DEBUG] 标准化原文: {ori_clean}")
            print(f"[DEBUG] 标准化识别: {gen_clean}")
            
            return wer
            
        except Exception as e:
            print(f"[ERROR] WER计算失败: {e}")
            return 1.0


def init_deterministic_rng(seed: int) -> torch.Generator:
    """初始化确定性随机数生成器"""
    print(f"[INFO] 设置确定性随机数种子: {seed}")
    
    # Python随机数
    random.seed(seed)
    
    # NumPy随机数
    np.random.seed(seed)
    
    # PyTorch随机数
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置确定性计算
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return torch.manual_seed(seed)


def save_rng_state():
    """保存所有随机数生成器状态"""
    return {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state(rng_state):
    """恢复所有随机数生成器状态"""
    random.setstate(rng_state['python_random_state'])
    np.random.set_state(rng_state['numpy_random_state'])
    torch.set_rng_state(rng_state['torch_random_state'])
    if rng_state['torch_cuda_random_state'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state['torch_cuda_random_state'])


def save_checkpoint_complete(
    checkpoint_path: Path, 
    global_step: int, 
    epoch: int, 
    data_index: int,
    model, 
    tokenizer, 
    optimizer, 
    scheduler=None,
    replay_buffer=None,
    additional_state=None
):
    """完整的检查点保存函数"""
    save_dir = checkpoint_path / f"step_{global_step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[INFO] 保存检查点到: {save_dir}")
        
        # 1. 保存模型和tokenizer
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("✅ 模型和tokenizer已保存")

        # 2. 保存优化器状态
        optimizer_path = save_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        print("✅ 优化器状态已保存")
        
        # 3. 保存学习率调度器（如果存在）
        if scheduler is not None:
            scheduler_path = save_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
            print("✅ 学习率调度器已保存")
        
        # 4. 保存随机数状态
        rng_state = save_rng_state()
        rng_path = save_dir / "rng_state.pt"
        torch.save(rng_state, rng_path)
        print("✅ 随机数状态已保存")
        
        # 5. 保存训练状态
        training_state = {
            'epoch': epoch,
            'global_step': global_step,
            'data_index': data_index,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        
        # 添加额外状态
        if additional_state:
            training_state.update(additional_state)
            
        state_path = save_dir / "training_state.json"
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(training_state, f, indent=2)
        print("✅ 训练状态已保存")
        
        # 6. 保存replay buffer（可选）
        if replay_buffer is not None and hasattr(replay_buffer, '__len__') and len(replay_buffer) > 0:
            buffer_path = save_dir / "replay_buffer.pt"
            torch.save(replay_buffer, buffer_path)
            print("✅ Replay buffer已保存")
        
        print(f"💾 检查点保存成功: {save_dir} (epoch={epoch}, step={global_step}, data_index={data_index})")
        return True
        
    except Exception as e:
        print(f"[ERROR] 保存检查点失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_checkpoint_complete(
    checkpoint_dir: str, 
    model, 
    tokenizer, 
    optimizer, 
    scheduler=None,
    total_data_len=None
):
    """完整的检查点加载函数"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"[ERROR] 检查点不存在: {checkpoint_path}")
        return 0, 0, 0, model, tokenizer, None
    
    try:
        print(f"[INFO] 从检查点恢复: {checkpoint_path}")
        
        # 1. 加载模型
        model = PeftModel.from_pretrained(model, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print("✅ 模型和tokenizer已加载")
        
        # 2. 加载优化器状态
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path))
            print("✅ 优化器状态已恢复")
        else:
            print("⚠️ 未找到优化器状态，将使用初始状态")
        
        # 3. 加载学习率调度器
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            if scheduler_path.exists():
                scheduler.load_state_dict(torch.load(scheduler_path))
                print("✅ 学习率调度器已恢复")
            else:
                print("⚠️ 未找到调度器状态")
        
        # 4. 加载随机数状态
        rng_path = checkpoint_path / "rng_state.pt"
        if rng_path.exists():
            rng_state = torch.load(rng_path)
            load_rng_state(rng_state)
            print("✅ 随机数状态已恢复，确保生成数据的一致性")
        else:
            print("⚠️ 未找到随机数状态，随机性可能不一致")
        
        # 5. 加载训练状态
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r', encoding='utf-8') as f:
                training_state = json.load(f)
            
            epoch = training_state.get('epoch', 0)
            global_step = training_state.get('global_step', 0)
            data_index = training_state.get('data_index', 0)
            learning_rate = training_state.get('learning_rate', None)
            
            if learning_rate and optimizer:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                print(f"✅ 学习率已恢复: {learning_rate}")
            
            print(f"✅ 训练状态已恢复: epoch={epoch}, global_step={global_step}, data_index={data_index}")
        else:
            # 从目录名推断
            if 'step_' in checkpoint_path.name:
                global_step = int(checkpoint_path.name.split('step_')[1])
                epoch = 0  # 无法准确推断，从0开始
                data_index = 0
            else:
                epoch, global_step, data_index = 0, 0, 0
            print(f"⚠️ 从目录名推断状态: global_step={global_step}")
        
        # 6. 加载replay buffer（可选）
        buffer_path = checkpoint_path / "replay_buffer.pt"
        replay_buffer = None
        if buffer_path.exists():
            try:
                replay_buffer = torch.load(buffer_path)
                print("✅ Replay buffer已恢复")
            except Exception as e:
                print(f"⚠️ Replay buffer加载失败: {e}")
        
        return epoch, global_step, data_index, model, tokenizer, replay_buffer
        
    except Exception as e:
        print(f"[ERROR] 加载检查点失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, model, tokenizer, None


def verify_checkpoint(checkpoint_dir: str):
    """验证检查点的完整性"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return False, "检查点目录不存在"
    
    required_files = [
        "config.json",
        "adapter_model.bin",  # LoRA权重
        "optimizer.pt",
        "training_state.json"
    ]
    
    optional_files = [
        "scheduler.pt",
        "rng_state.pt",
        "replay_buffer.pt"
    ]
    
    missing_required = []
    missing_optional = []
    
    for file in required_files:
        if not (checkpoint_path / file).exists():
            missing_required.append(file)
    
    for file in optional_files:
        if not (checkpoint_path / file).exists():
            missing_optional.append(file)
    
    if missing_required:
        return False, f"缺少必需文件: {missing_required}"
    
    status = "检查点完整"
    if missing_optional:
        status += f"，缺少可选文件: {missing_optional}"
    
    return True, status


def load_whisper(model_name="openai/whisper-base", device="cuda:4"):
    """加载Whisper模型"""
    print(f"[INFO] 加载Whisper模型: {model_name}, 设备: {device}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    print(f"[INFO] Whisper模型加载完成")
    return processor, model


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    bf16: bool = True,
    device_map=None,
    use_lora: bool = False,
    lora_config: Optional[LoraConfig] = None,
) -> tuple[MiniCPMO, PreTrainedTokenizer]:
    """加载MiniCPM模型"""
    from peft import get_peft_model
    print(f"[INFO] 加载MiniCPM模型: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    model = MiniCPMO.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation='sdpa',
        torch_dtype=torch.float16
    )
    if use_lora:
        assert lora_config is not None, "LoRA config required"
        model = get_peft_model(model, lora_config)
        print(f"[INFO] LoRA配置应用完成")
    return model, tokenizer


def load_data_from_json(json_file_path: str) -> list:
    """从JSON文件加载数据并构建prompt"""
    prompts = []
    try:
        print(f"[INFO] 从JSON文件加载数据: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            text_id = item["id"]
            text_content = item["text"]
            labels = item["labels"]
            
            # 解析标签 - 改进正则表达式
            structure = re.search(r"1\.\s*Structure:\s*(.*?)(?:\n|$)", labels, re.MULTILINE)
            emotion = re.search(r"2\.\s*Emotion:\s*(.*?)(?:\n|$)", labels, re.MULTILINE)
            speech_speed = re.search(r"3\.\s*Speech Speed:\s*(.*?)(?:\n|$)", labels, re.MULTILINE)
            tone = re.search(r"4\.\s*Tone:\s*(.*?)(?:\n|$)", labels, re.MULTILINE)
            
            structure = structure.group(1).strip() if structure else ""
            emotion = emotion.group(1).strip() if emotion else ""
            speech_speed = speech_speed.group(1).strip() if speech_speed else ""
            tone = tone.group(1).strip() if tone else ""
            
            # 构建指令
            instruction = f"Please express the sentence '{text_content}' with a structure that is {structure}, " \
                          f"emotionally {emotion}, at a speech speed of {speech_speed}, and in a {tone} tone, " \
                          f"voiced by a middle-aged woman."

            prompts.append({
                "prompt": instruction,
                "id": text_id,
                "original_text": text_content,
                "target_labels": {
                    "structure": structure.lower(),
                    "emotion": emotion.lower(),
                    "speech_speed": speech_speed.lower().replace(" ", "_"),
                    "tone": tone.lower()
                }
            })
            
        print(f"[INFO] 从JSON文件加载数据完成，共{len(prompts)}条记录")
        return prompts
        
    except Exception as e:
        print(f"[ERROR] 从JSON文件加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def pad_sequences_to_same_length(sequences: list, pad_token_id: int, max_length: int = None) -> torch.Tensor:
    """将序列填充到相同长度"""
    if not sequences:
        return torch.empty(0)
    
    # 如果没有指定最大长度，使用序列中的最大长度
    if max_length is None:
        max_length = max(seq.size(1) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        current_length = seq.size(1)
        if current_length < max_length:
            # 右侧填充
            padding = torch.full((seq.size(0), max_length - current_length), 
                               pad_token_id, dtype=seq.dtype, device=seq.device)
            padded_seq = torch.cat([seq, padding], dim=1)
        else:
            # 截断到最大长度
            padded_seq = seq[:, :max_length]
        padded_sequences.append(padded_seq)
    
    return torch.cat(padded_sequences, dim=0)


@torch.no_grad()
def rollout_single(
    model: MiniCPMO,
    tokenizer: PreTrainedTokenizer,
    text: str,
    id: str,
    target_labels: dict,
    original_text: str,
    whisper_processor,
    whisper_model,
    qwen_analyzer: QwenAudioAnalyzer,
    wer_calculator: WERCalculator,
    num_rollouts: int,
    device: str,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """单个样本的rollout生成"""

    model.eval()
    model = model.to(device)
    
    # 提取纯ID（去掉路径和扩展名）
    pure_id = os.path.splitext(os.path.basename(id))[0]

    chat_messages = [{'role':'user', 'content':[text]}]
    
    # 为每个rollout生成不同的音频文件和计算奖励
    wers_list = []
    tag_rewards = []
    all_sequences = []
    all_action_masks = []
    all_results = []
    successful_rollouts = 0
    
    for i in range(num_rollouts):
        print(f"[INFO] 处理 rollout {i+1}/{num_rollouts} for sample {pure_id}")
        
        # 生成唯一的音频文件名
        audio_filename = f"{pure_id}_rollout_{i}_{uuid.uuid4().hex[:8]}.wav"
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)
        
        # 使用MiniCPM生成语音
        try:
            print(f"[DEBUG] 开始生成音频: {audio_path}")
            result = model.chat(
                msgs=chat_messages,
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=128,
                use_tts_template=True,
                generate_audio=True,
                temperature=temperature,
                output_audio_path=audio_path,
            )
            
            print(f"[INFO] 生成音频成功: {audio_path}")
            all_results.append(result)
            
            # 准备模型输入用于获取序列
            model_inputs = model._prepare_inputs(
                msgs=chat_messages,
                tokenizer=tokenizer,
                sampling=True,
                use_tts_template=True,
                generate_audio=True,
                temperature=temperature,
                max_inp_length=256
            )
            
            pad_token_id = tokenizer.eos_token_id
            generation_config = GenerationConfig(
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                pad_token_id=pad_token_id,
            )
            
            _, outputs = model.generate(
                **model_inputs,
                generation_config=generation_config,
                tokenizer=tokenizer
            )
            
            # 获取序列信息
            sequence_ids = outputs.sequences
            action_mask = torch.ones_like(sequence_ids, dtype=torch.bool)
            action_mask[sequence_ids == pad_token_id] = False
            action_mask[sequence_ids == 0] = False
            action_mask = action_mask[:, 1:]
            
            all_sequences.append(sequence_ids)
            all_action_masks.append(action_mask)
            successful_rollouts += 1
            
            # 使用Whisper转录音频
            print(f"[DEBUG] 开始Whisper转录...")
            gen_text = wer_calculator.transcribe_with_whisper(
                audio_path, whisper_processor, whisper_model, device="cuda:4"
            )
            
            # 计算WER
            print(f"[DEBUG] 计算WER...")
            wer = wer_calculator.compute_wer(f"{pure_id}_rollout_{i}", original_text, gen_text)
            wers_list.append(torch.tensor([wer], dtype=torch.float))

            # 计算标签匹配奖励
            print(f"[DEBUG] 计算标签匹配奖励...")
            reward = qwen_analyzer.compute_tag_match_reward(audio_path, target_labels)
            tag_rewards.append(reward)
            
        except Exception as e:
            print(f"[ERROR] 音频生成失败 rollout {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if successful_rollouts == 0:
        print(f"[ERROR] 所有rollout都失败，跳过样本 {pure_id}")
        return None, None, None, None

    # 处理不足的rollout - 使用最后一个成功的结果复制
    if successful_rollouts < num_rollouts:
        print(f"[WARNING] 只有 {successful_rollouts}/{num_rollouts} rollout成功，使用最后成功的结果填充")
        
        # 获取最后一个成功的结果
        last_sequence = all_sequences[-1]
        last_action_mask = all_action_masks[-1]
        last_wer = wers_list[-1]
        last_reward = tag_rewards[-1]
        
        # 复制直到达到所需的rollout数量
        while len(all_sequences) < num_rollouts:
            all_sequences.append(last_sequence.clone())
            all_action_masks.append(last_action_mask.clone())
            wers_list.append(last_wer.clone())
            tag_rewards.append(last_reward)

    # 填充序列到相同长度
    print(f"[DEBUG] 填充序列到相同长度...")
    try:
        # 先检查序列长度
        seq_lengths = [seq.size(1) for seq in all_sequences[:num_rollouts]]
        print(f"[DEBUG] 序列长度: {seq_lengths}")
        
        # 使用填充函数
        sequence_ids = pad_sequences_to_same_length(
            all_sequences[:num_rollouts], 
            tokenizer.eos_token_id
        )
        
        # 对action_mask也进行相同的填充处理
        action_mask = pad_sequences_to_same_length(
            all_action_masks[:num_rollouts], 
            False  # action_mask用False填充
        )
        
        print(f"[DEBUG] 填充后序列形状: {sequence_ids.shape}")
        print(f"[DEBUG] 填充后action_mask形状: {action_mask.shape}")
        
    except Exception as e:
        print(f"[ERROR] 序列填充失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # 计算最终奖励
    wers_ = torch.stack(wers_list[:num_rollouts]).view(-1)
    returns = compute_reward(wers_, tag_rewards[:num_rollouts]).view(-1)

    # 记录到wandb
    wandb.log({
        "reward_mean": returns.mean().item(),
        "reward_max": returns.max().item(),
        "reward_min": returns.min().item(),
        "WER_mean": wers_.mean().item(),
        "tag_reward_mean": sum(tag_rewards[:num_rollouts]) / len(tag_rewards[:num_rollouts])
    })

    print(f"[INFO] 样本 {pure_id} 结果:")
    print(f"[INFO]   WER scores: {wers_}")
    print(f"[INFO]   Tag rewards: {tag_rewards[:num_rollouts]}")
    print(f"[INFO]   Final rewards: {returns}")

    return sequence_ids, returns, action_mask, all_results[0] if all_results else None


def compute_reward(wer_scores, tag_rewards):
    """计算综合奖励"""
    wer_complement = 1.0 - wer_scores  # WER 补值为 (1 - wer)，即正确率   

    rewards = 0.3 * wer_complement  # WER权重30%

    if tag_rewards is not None:
        tag_rewards_tensor = torch.tensor(tag_rewards, dtype=torch.float, device=wer_scores.device)
        rewards += 0.7 * tag_rewards_tensor  # 标签匹配权重70%

    rewards = torch.clamp(rewards, min=0.0, max=1.0)
    return rewards


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算群组优势"""
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    """从logits计算序列log概率"""
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: MiniCPMO,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """计算序列的log概率"""
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
    """读取JSONL文件"""
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

 
def main():
    print("🚀 启动MiniCPM GRPO训练...")
    
    # 设备配置
    model_device = torch.device("cuda:6")       # MiniCPM主模型
    ref_device = torch.device("cuda:7")         # MiniCPM参考模型  
    whisper_device = torch.device("cuda:4")     # Whisper模型
    # qwen_device = cuda:0 在qwen_audio_service.py中设置

    # 训练参数
    num_epochs = 10
    seed = 42
    wandb_project = "None"  # 设置wandb项目名
    model_name = "MiniCPM-o"

    # LoRA 配置
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

    # 其他训练参数
    checkpoint_path = Path("the path of checkpoint")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_interval = 20
    train_batch_size = 1
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2
    group_size = 4
    epochs_per_step = 1
    max_norm = 1.0
    max_length = 256
    top_p = 1.0
    temperature = 1.0

    # 🔥 恢复选项
    resume_from_checkpoint = "Set the checkpoint path to be restored"  # 设置要恢复的检查点路径
    # resume_from_checkpoint = None  # 如果要从头开始训练

    # 验证检查点（如果存在）
    if resume_from_checkpoint:
        is_valid, status = verify_checkpoint(resume_from_checkpoint)
        if is_valid:
            print(f"✅ 检查点验证通过: {status}")
        else:
            print(f"❌ 检查点验证失败: {status}")
            print("将从头开始训练...")
            resume_from_checkpoint = None

    print(f"[INFO] 初始化确定性随机数种子: {seed}")
    init_deterministic_rng(seed)

    print("🔧 初始化分析器和计算器...")
    
    # 初始化分析器
    qwen_analyzer = QwenAudioAnalyzer()  # 使用subprocess版本
    wer_calculator = WERCalculator()

    # 加载Whisper模型
    print(f"[INFO] 加载Whisper模型到设备: {whisper_device}")
    whisper_processor, whisper_model = load_whisper("the path of whisper-large-v3", device=whisper_device)
    print(f"✅ Whisper模型加载完成")

    # 加载参考模型
    print(f"[INFO] 加载参考模型到设备: {ref_device}")
    reference_model, _ = load_model(model_name)
    reference_model.to(ref_device) 
    reference_model.eval()
    print(f"✅ 参考模型加载完成")

    # 加载训练模型
    print(f"[INFO] 加载训练模型到设备: {model_device}")
    model, tokenizer = load_model(
        model_name,
        use_lora=True,
        lora_config=lora_config
    )
    model.to(model_device)
    model.print_trainable_parameters()
    model.init_tts()
    model.tts.float()
    print(f"✅ 训练模型加载完成")

    # 创建优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)  # 每100步衰减学习率
    
    # 数据加载
    print("📊 加载训练数据...")
    pad_token_id = tokenizer.eos_token_id
    
    # 直接从JSON文件加载数据
    json_path = "the path of processed_labels_3_formatted.json"
    prompts = load_data_from_json(json_path)
    
    if not prompts:
        print("❌ 数据加载失败，退出程序")
        return
    
    print(f"✅ 数据加载完成，共 {len(prompts)} 个样本")
    
    # 创建有序数据迭代器
    data_iterator = OrderedDataIterator(prompts)
    
    # 🔥 恢复检查点
    start_epoch = 0
    global_step = 0
    start_data_index = 0
    old_replay_buffer = None
    
    if resume_from_checkpoint:
        result = load_checkpoint_complete(
            resume_from_checkpoint, 
            model, 
            tokenizer, 
            optimizer, 
            scheduler,
            len(prompts)
        )
        start_epoch, global_step, start_data_index, model, tokenizer, old_replay_buffer = result
        
        # 设置数据迭代器的起始位置
        data_iterator.set_current_index(start_data_index)
        print(f"✅ 数据迭代器已设置到索引: {start_data_index}")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    replay_buffer = ReplayBuffer()
    if old_replay_buffer is not None:
        # 可以选择是否使用旧的replay buffer
        print("[INFO] 发现旧的replay buffer，重新开始buffer")
        # replay_buffer = old_replay_buffer  # 如果要使用旧buffer，取消注释这行
    
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    # 初始化wandb
    wandb.init(mode="disabled")
    print("[INFO] WandB已禁用")

    print("🎯 开始训练循环...")
    print(f"[INFO] 从epoch {start_epoch}, global_step {global_step}, data_index {start_data_index}开始")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"🔄 Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        epoch_rollout_returns = []
        samples_processed_this_epoch = 0

        # 每个epoch重置数据迭代器（除了第一个恢复的epoch）
        if epoch > start_epoch:
            data_iterator.reset()
            print(f"[INFO] Epoch {epoch + 1}: 数据迭代器已重置")

        # 按顺序处理数据
        try:
            while not data_iterator.is_finished():
                sample_data, current_data_index = next(data_iterator)
                
                print(f"\n{'─'*40}")
                print(f"📝 Epoch {epoch+1}, Global Step: {global_step}")
                print(f"📍 Data Index: {current_data_index}/{len(prompts)}")
                print(f"{'─'*40}")
                
                sample = sample_data
                texts = [sample["prompt"]]
                ids = [sample["id"]]
                target_labels_list = [sample["target_labels"]]
                original_texts = [sample["original_text"]]

                print(f"[INFO] 处理样本: {ids[0]}")
                print(f"[INFO] 指令: {texts[0][:100]}...")
                print(f"[INFO] 目标标签: {target_labels_list[0]}")

                rollout_returns = []
                replay_buffer.clear()

                # Rollout阶段
                print(f"[INFO] 开始Rollout阶段...")
                with torch.no_grad():
                    rollout_result = rollout_single(
                        model.to(model_device),
                        tokenizer,
                        text=texts[0],
                        id=ids[0],
                        target_labels=target_labels_list[0],
                        original_text=original_texts[0],
                        whisper_processor=whisper_processor,
                        whisper_model=whisper_model,
                        qwen_analyzer=qwen_analyzer,
                        wer_calculator=wer_calculator,
                        num_rollouts=group_size,
                        device=model_device,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    if rollout_result[0] is None:
                        print("⚠️  Rollout失败，跳过此样本")
                        global_step += 1
                        continue

                    sequence_ids, returns, action_mask, completions = rollout_result

                    sequence_ids = sequence_ids.to(model_device)
                    returns = returns.to(model_device)
                    action_mask = action_mask.to(model_device)

                    rollout_returns.append(returns.cpu())
                    epoch_rollout_returns.append(returns.cpu())
                    
                    advantages = group_advantages(returns).view(-1).to(model_device)
                    
                    if torch.all(advantages == 0):
                        print("⚠️  所有advantages为0，跳过此batch")
                        global_step += 1
                        continue
                        
                    returns = returns.view(-1) 
                    attention_mask = sequence_ids != torch.full_like(sequence_ids, pad_token_id, device=sequence_ids.device)

                    print("[INFO] 计算log probabilities...")
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
                    replay_buffer.append(experience)

                torch.cuda.empty_cache()
                
                if rollout_returns:
                    episode_return_sum = torch.stack(rollout_returns).sum()
                    print(f"[INFO] Step {global_step} returns: {episode_return_sum:.4f}")
                    wandb.log({"returns": episode_return_sum, "step": global_step})
                else:
                    print("⚠️  无有效returns")
                    global_step += 1
                    continue

                # 策略更新阶段
                print("[INFO] 开始策略更新...")
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
                        exp = exp.to(next(model.parameters()).device)
                        optimizer.zero_grad()

                        log_probs = sequences_log_probs(
                            model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                        )

                        loss, kl = objective(log_probs=log_probs, experience=exp)
                        
                        if not loss.isfinite() or not kl.isfinite():
                            print(f"⚠️  Loss not finite, skipping: loss={loss}, kl={kl}")
                            continue

                        loss.backward()
                        total_loss += loss.item()
                        total_steps += 1
                        
                        grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                        
                        wandb.log({
                            "loss": loss.item(),
                            "kl": kl.item(), 
                            "grad_norm": grad_norm.item(),
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "step": global_step
                        })

                        optimizer.step()
                        
                    if total_steps > 0:
                        avg_loss = total_loss / total_steps
                        print(f"[INFO] Epoch {step_epoch+1} avg loss: {avg_loss:.6f}")
                    else:
                        avg_loss = 0.0
                        print("⚠️  No valid training steps")

                # 更新学习率调度器
                scheduler.step()
                
                # 🔥 保存检查点（使用完整版本）
                if (
                    checkpoint_path is not None
                    and checkpoint_interval is not None
                    and (global_step + 1) % checkpoint_interval == 0
                ):
                    additional_state = {
                        'samples_processed_this_epoch': samples_processed_this_epoch + 1,
                        'total_samples': len(prompts),
                    }
                    
                    success = save_checkpoint_complete(
                        checkpoint_path, 
                        global_step, 
                        epoch, 
                        current_data_index + 1,  # 下一个要处理的数据索引
                        model, 
                        tokenizer, 
                        optimizer, 
                        scheduler,
                        replay_buffer,
                        additional_state
                    )
                    
                    if not success:
                        print(f"[WARNING] 保存检查点失败，但继续训练...")
                    
                global_step += 1
                samples_processed_this_epoch += 1
                
        except StopIteration:
            print(f"[INFO] Epoch {epoch + 1} 所有数据处理完成")
            
        # Epoch结束统计
        if epoch_rollout_returns:
            epoch_avg_return = torch.cat(epoch_rollout_returns).mean().item()
            print(f"\n🏁 Epoch {epoch+1} 完成:")
            print(f"   处理样本数: {samples_processed_this_epoch}")
            print(f"   平均奖励: {epoch_avg_return:.4f}")
            print(f"   当前学习率: {scheduler.get_last_lr()[0]:.2e}")
            
            wandb.log({
                "epoch": epoch + 1,
                "epoch_avg_return": epoch_avg_return,
                "samples_processed": samples_processed_this_epoch,
                "learning_rate": scheduler.get_last_lr()[0],
            })
        else:
            print(f"\n⚠️  Epoch {epoch+1} 无有效数据")

    # 保存最终模型
    if checkpoint_path is not None:
        final_dir = checkpoint_path / "final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        # 保存最终状态
        final_state = {
            'completed_epochs': num_epochs,
            'total_steps': global_step,
            'final_learning_rate': optimizer.param_groups[0]['lr'],
            'training_completed': True
        }
        
        with open(final_dir / "final_state.json", 'w', encoding='utf-8') as f:
            json.dump(final_state, f, indent=2)
            
        print(f"🎉 最终模型已保存: {final_dir}")

    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)
    print(f"📁 输出文件位置:")
    print(f"   🎵 音频文件: {AUDIO_OUTPUT_DIR}")
    print(f"   📊 WER结果: {WER_RESULT_CSV}")
    print(f"   🏷️  标签分析: {AUDIO_LABEL_RESULT_JSON}")
    print(f"   💾 模型检查点: {checkpoint_path}")


if __name__ == "__main__":
    main()