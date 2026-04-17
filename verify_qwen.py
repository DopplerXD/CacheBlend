#!/usr/bin/env python3
"""验证 Qwen2.5-1.5B 模型可用性"""

import sys
sys.path.insert(0, '/root/github/CacheBlend')

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/models/Qwen2.5-1.5B"

def main():
    print(f"加载 tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Tokenizer 加载成功!")

    print(f"加载模型 from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("模型加载成功!")

    prompt = "你好，请介绍一下你自己"
    print(f"\n输入: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"输出: {response}")
    print("\n✅ Qwen2.5-1.5B 模型验证通过!")

if __name__ == "__main__":
    main()