#!/usr/bin/env python3
import os
import sys
import subprocess
import zipfile
from datetime import datetime

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 命令失败: {cmd}")
        print(f"错误: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("📦 诊断Agent版本备份工具")
    print("=" * 50)
    
    try:
        commit_hash = run_command("git rev-parse --short HEAD")
        commit_message = run_command("git log -1 --format=%s")
        branch = run_command("git rev-parse --abbrev-ref HEAD")
        
        print(f"当前分支: {branch}")
        print(f"当前提交: {commit_hash}")
        print(f"提交信息: {commit_message}")
        print()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"diagnosis_agent_{commit_hash}_{timestamp}"
        
        print(f"创建补丁文件...")
        patch_file = f"{backup_name}.patch"
        run_command(f"git format-patch --stdout HEAD > {patch_file}")
        
        print(f"创建完整打包...")
        zip_file = f"{backup_name}.zip"
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk('.'):
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'data', 'output']]
                for file in files:
                    if file.endswith(('.pyc', '.pyo')) or file.startswith('.'):
                        continue
                    if '/.' in root:
                        continue
                    filepath = os.path.join(root, file)
                    zf.write(filepath)
        
        print(f"\n✅ 备份完成！")
        print(f"补丁文件: {patch_file}")
        print(f"打包文件: {zip_file}")
        print(f"大小: {os.path.getsize(zip_file) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ 备份失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()