import os
import re

def clean_filenames(folder):
    # 需要去掉的固定模式
    pattern = r"\[ 微信号：itcodeba  \]\(更多it资源站 www\.dashendao\.com\)"
    
    for root, dirs, files in os.walk(folder):
        for filename in files:
            old_path = os.path.join(root, filename)
            # 用正则去掉广告部分
            new_filename = re.sub(pattern, "", filename)
            # 去掉多余空格
            new_filename = new_filename.strip()
            new_path = os.path.join(root, new_filename)

            if old_path != new_path:
                print(f"重命名: {old_path} -> {new_path}")
                os.rename(old_path, new_path)

if __name__ == "__main__":
    folder = r"Langchain\01-Langchain"  # 修改为你的目录
    clean_filenames(folder)
