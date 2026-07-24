"""Diagnosis Agent 入口模块

支持 python -m diagnosis_agent 运行方式
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diagnosis_agent.cli import main

if __name__ == '__main__':
    main()