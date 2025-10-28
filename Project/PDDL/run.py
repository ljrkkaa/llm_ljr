import subprocess
import sys

# domain = sys.argv[1]
# problem = sys.argv[2]

cmd = [
    "ff-v2.1.exe",
    "-p", "test/",
    "-o", "cake-domain-plus.pddl",
    "-f", "cake-problem-plus.pddl",
    "-s", "0"   # 非成本搜索
]

subprocess.run(cmd)
