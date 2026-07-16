# 确保项目根目录在 sys.path 中，使 pytest 能直接运行
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
