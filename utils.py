from modelscope.msdatasets import MsDataset
import json
import random
from pathlib import Path
import shutil


def delete_cache(project_root_path='.'):
    # 将传入的路径转换为Path对象，确保路径操作的兼容性
    root_path = Path(project_root_path)
    # 使用rglob查找所有__pycache__目录
    pycache_dirs = root_path.rglob('__pycache__')
    # 遍历找到的__pycache__目录列表并删除它们
    for pycache_dir in pycache_dirs:
        print(f"Deleting: {pycache_dir}")
        shutil.rmtree(pycache_dir)
    print("All __pycache__ directories have been deleted.")


def predict(messages, model, tokenizer):
    device = "cuda"
    # 关闭 gradient checkpointing 以避免 generate 时冲突
    model.gradient_checkpointing_disable()

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 生成完后，重新启用 gradient checkpointing（方便后续再训练或别的操作）
    model.gradient_checkpointing_enable()
    return response


def load_dataset():
    # 设置随机种子以确保可重复性
    random.seed(42)
    # 加载数据集
    ds = MsDataset.load('krisfu/delicate_medical_r1_data', subset_name='default', split='train')
    # 将数据集转换为列表
    data_list = list(ds)
    # 随机打乱数据
    random.shuffle(data_list)
    # 计算分割点
    split_idx = int(len(data_list) * 0.9)
    # 分割数据
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    # 保存训练集
    with open('train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    # 保存验证集
    with open('val.jsonl', 'w', encoding='utf-8') as f:
        for item in val_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"数据集已分割完成：")
    print(f"训练集大小：{len(train_data)}")
    print(f"验证集大小：{len(val_data)}")