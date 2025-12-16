from huggingface_hub import hf_hub_download
import os

# 配置参数
REPO_ID = "allenai/c4"
SUBFOLDER = "en"
SAVE_DIR = "./c4"
os.makedirs(SAVE_DIR, exist_ok=True)

# 文件列表（完整的 train 和 validation）
train_files = [f"c4-train.{i:05d}-of-01024.json.gz" for i in range(1024)]
val_files = [f"c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]
all_files = train_files + val_files

print(f"准备下载 allenai/c4/en 中的 {len(all_files)} 个数据文件...")

# 开始下载
for filename in all_files:
    print(f" 正在下载: {filename}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        subfolder=SUBFOLDER,
        repo_type="dataset",
        local_dir=SAVE_DIR,
        local_dir_use_symlinks=False
    )
    print(f"完成: {path}\n")

print(" 所有文件下载完成！")
