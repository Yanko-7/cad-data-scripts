#!/bin/bash

# 1. 定义处理单个文件的函数
process_file() {
    f="$1"
    # 提取文件名中的数字部分
    num=$(echo "$f" | cut -d'_' -f2 | sed 's/^0*//')
    
    # 补0逻辑
    [ -z "$num" ] && num=0

    target="chunk$num"
    
    # 创建目录并解压
    mkdir -p "$target"
    # 使用 -aos 跳过已存在的文件，避免并行写入冲突（可选）
    7z x "$f" -o"$target" > /dev/null
}

# 2. 导出函数，使其对 parallel 可见
export -f process_file

# 3. 使用 parallel 并行执行
# -j 80: 指定 80 个并行任务
# --bar: 显示进度条 (如果你的 parallel 版本太老不支持，可以去掉这个参数)
ls abc_*.7z | parallel -j 80 --bar process_file {}
