import sys
sys.path.append("the path of cloning Qwen2-Audio-main")

import os
import json
import re
import torch
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def analyze_audio(audio_path: str) -> dict:
    """
    分析音频文件，返回结构、情感、语速、语调标签
    """
    try:
        # 模型配置
        model_path = "Qwen2-Audio"
        device = "cuda:0"  # 按照你的要求使用cuda:
        
        print(f"[DEBUG] 开始分析音频: {audio_path}")
        print(f"[DEBUG] 使用设备: {device}")
        
        # 检查音频文件
        if not os.path.exists(audio_path):
            print(f"[ERROR] 音频文件不存在: {audio_path}")
            return {
                "structure": "-1",
                "emotion": "-1", 
                "speech_speed": "-1",
                "tone": "-1"
            }
            
        if os.path.getsize(audio_path) == 0:
            print(f"[ERROR] 音频文件为空: {audio_path}")
            return {
                "structure": "-1",
                "emotion": "-1", 
                "speech_speed": "-1",
                "tone": "-1"
            }

        structure_label_map = {
            'Introduction': 0, 'Background': 1, 'Argument': 2, 'Conclusion': 3,
            'Transition': 4, 'Example': 5, 'Problem Statement': 6, 'Methodology': 7,
            'Data Analysis': 8, 'Summary': 9, 'Structure': 10, 'Dialogue': 11, 'Command': 12,
            'Description': 13
        }

        emotion_label_map = {
            'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Excited': 3, 'Worried': 4,
            'Desperate': 5, 'Angry': 6, 'Expectant': 7, 'Confused': 8, 'Optimistic': 9,
            'Anxious': 10, 'Defiant': 11, 'Frustrated': 12, 'Uneasy': 13, 'Astonished': 14,
            'Defensive': 15, 'Determined': 16, 'Urgent': 17, 'Surprised': 18, 'Curious': 19,
            'Suspicious': 20, 'Melancholy': 21, 'Dramatic': 22, 'Sad': 23, 'Stern': 24
        }

        speech_speed_map = {
            'Extremely Slow': 0, 'Slow': 1, 'Medium Slow': 2, 'Medium Fast': 3, 'Extremely Fast': 4,
            'Medium fast': 3
        }

        tone_label_map = {
            'Declarative': 0, 'Interrogative': 1, 'Imperative': 2, 'Exclamation': 3,
            'Rhetorical Question': 4, 'Command': 5
        }

        print("[DEBUG] 加载模型...")
        
        # 加载模型和处理器 - 按照audio_label_reason.py的方式
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        # model = Qwen2AudioForConditionalGeneration.from_pretrained(
        #     model_path, device_map="auto", local_files_only=True
        # )
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True
        )

        sampling_rate = processor.feature_extractor.sampling_rate
        print(f"[DEBUG] 采样率: {sampling_rate}")

        # 构造指令文本 - 与audio_label_reason.py保持一致
        structure_list = ", ".join(structure_label_map.keys())
        emotion_list = ", ".join(emotion_label_map.keys())
        speech_speed_list = ", ".join(speech_speed_map.keys())
        tone_list = ", ".join(tone_label_map.keys())

        instruction_text = (
            f"Please classify the audio into the following categories:\n\n"
            f"Structure: one of [{structure_list}]\n"
            f"Emotion: one of [{emotion_list}]\n"
            f"Speech Speed: one of [{speech_speed_list}]\n"
            f"Tone: one of [{tone_list}]\n\n"
            f"Respond strictly in JSON format using double quotes.\n\n"
            f"Output format:\n"
            f"{{\n"
            f"  \"Structure\": \"<label>\",\n"
            f"  \"Emotion\": \"<label>\",\n"
            f"  \"Speech Speed\": \"<label>\",\n"
            f"  \"Tone\": \"<label>\"\n"
            f"}}"
        )

        # 构造对话结构 - 完全按照audio_label_reason.py的方式
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": f"file://{audio_path}"},
                {"type": "text", "text": instruction_text}
            ]}
        ]

        # 应用 chat template
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        print("[DEBUG] 加载音频数据...")
        # 加载音频为 float32 np.array - 完全按照audio_label_reason.py的方式
        audio_data, _ = librosa.load(audio_path, sr=sampling_rate)
        audio_data = np.asarray(audio_data, dtype=np.float32).copy(order="C")
        print(f"[DEBUG] Audio type: {type(audio_data)}, shape: {audio_data.shape}")

        # 构造输入（使用新字段名 audio）- 完全按照audio_label_reason.py的方式
        inputs = processor(
            text=[text],
            audio=[audio_data],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )

        # 移动 input_ids 到显卡
        inputs.input_ids = inputs.input_ids.to(model.device)

        print("[DEBUG] 开始模型推理...")
        # 推理生成
        generate_ids = model.generate(**inputs, max_new_tokens=128)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        # 解码结果
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(f"[DEBUG] 原始响应: {response}")

        # 处理输出 - 按照audio_label_reason.py的解析方式
        try:
            response_cleaned = response.replace("'", '"')
            response_cleaned = re.sub(r"^[^({\[]*", "", response_cleaned).strip()
            parsed = json.loads(response_cleaned)
            
            print(f"[DEBUG] 解析成功: {parsed}")
            
            # 转换为小写并标准化格式
            result = {
                "structure": parsed.get("Structure", "-1").lower(),
                "emotion": parsed.get("Emotion", "-1").lower(),
                "speech_speed": parsed.get("Speech Speed", "-1").lower().replace(" ", "_"),
                "tone": parsed.get("Tone", "-1").lower()
            }
            
            print(f"[DEBUG] 标准化结果: {result}")
            return result
            
        except Exception as e:
            print(f"[ERROR] JSON解析失败: {e}")
            print(f"[DEBUG] 尝试解析的内容: {response_cleaned}")
            return {
                "structure": "-1",
                "emotion": "-1",
                "speech_speed": "-1",
                "tone": "-1"
            }

        # 清理GPU缓存
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[ERROR] 音频分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "structure": "-1",
            "emotion": "-1",
            "speech_speed": "-1",
            "tone": "-1"
        }

def main():
    if len(sys.argv) != 2:
        print("Usage: python qwen_audio_service.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"[INFO] 开始分析音频文件: {audio_path}")

    result = analyze_audio(audio_path)
    
    # 输出JSON结果，供main_grpo.py解析
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()