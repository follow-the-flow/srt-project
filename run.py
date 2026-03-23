"""
启动脚本 - 手势控制 QArm 演示程序
使用方法: python run.py [--camera 0]
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import main

if __name__ == "__main__":
    main()
