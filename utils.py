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
