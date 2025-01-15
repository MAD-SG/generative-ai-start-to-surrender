#!/usr/bin/env bash
#
# 用法：
#   ./fix_math_block_inplace.sh your_file.md
#
# 功能：
#   1) 先检查文件中的 $$ 是否两两匹配，若不匹配则报错并退出；
#   2) 若匹配，则执行插入空行逻辑并覆盖原文件。
#
# 说明：
#   - 为兼容一些旧版 awk（如 macOS 自带 awk），函数定义都放在脚本最前面。
#   - 建议先备份原文件，以免发生意外。

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "用法：$0 [要处理的 Markdown 文件]"
  exit 1
fi

file="$1"
tmpfile="$(mktemp)"

awk '
###############################################################################
# 先定义函数（在任何 BEGIN/END/动作段之前），以兼容旧版 awk
###############################################################################
function ensureEmptyLineBefore() {
  if (lastLineEmpty == 0) {
    print ""
  }
}

function ensureEmptyLineAfter() {
  print ""
  lastLineEmpty = 1
}

###############################################################################
# 第一阶段：读取所有行到数组，统计 $$ 是否匹配
###############################################################################
BEGIN {
  inMath = 0          # 标记是否处在公式环境中
  nlines = 0          # 文件总行数
}

{
  nlines++
  lines[nlines] = $0   # 存下当前行

  # 统计本行出现多少次 $$，用于配对检查
  text = $0
  count = 0
  while (match(text, /\$\$/)) {
    count++
    text = substr(text, RSTART + RLENGTH)
  }

  # 每出现一次 $$，翻转 inMath
  for (i = 1; i <= count; i++) {
    inMath = 1 - inMath
  }
}

END {
  # 如果 inMath != 0，说明 $$ 不匹配
  if (inMath != 0) {
    print "Error: Unmatched $$ detected! 请检查 $$ 是否成对匹配。"
    exit 1
  }

  #-----------------------
  # 若匹配，第二遍输出
  #-----------------------
  inMath = 0         # 重置多行公式标记
  lastLineEmpty = 1  # 标记上一行是否为空行（初始假设空行）

  for (l = 1; l <= nlines; l++) {
    line = lines[l]
    sub(/[ \t]+$/, "", line)  # 去掉行尾空格

    # 判断空行
    if (length(line) == 0) {
      print ""
      lastLineEmpty = 1
      continue
    }

    # 统计这一行出现多少次 $$
    text = line
    count = 0
    while (match(text, /\$\$/)) {
      count++
      text = substr(text, RSTART + RLENGTH)
    }

    # 如果当前处于多行公式块
    if (inMath == 1) {
      if (count == 0) {
        # 多行公式内容，原样输出
        print line
        lastLineEmpty = 0
      } else {
        # 行中又出现 $$，可能结束公式块
        if (count % 2 == 1) {
          # 出现奇数个 $$ => 公式块结束
          print line
          lastLineEmpty = 0
          inMath = 0
          ensureEmptyLineAfter()
        } else {
          # 出现偶数个 $$ => 比较罕见的情况（多行公式中有额外单行公式？）
          # 在此简单地原样输出
          print line
          lastLineEmpty = 0
        }
      }
      continue
    }

    #--------------------
    # 不处于多行公式时
    #--------------------
    if (count == 0) {
      # 没有 $$，普通行
      print line
      lastLineEmpty = 0
    } else {
      if (count % 2 == 0) {
        # 行中出现 2n 个 $$ => 把它看作一个(或多个)单行公式
        ensureEmptyLineBefore()
        print line
        ensureEmptyLineAfter()
      } else {
        # 行中出现奇数个 $$ => 开始（或开始+结束）的多行公式
        ensureEmptyLineBefore()
        print line
        lastLineEmpty = 0
        inMath = 1

        # 特殊情况：一行出现 3 个 $$ => 实际上开始又结束
        # 这里如有需要，可再次翻转
        if (count > 1 && count % 2 == 1) {
          # 比如 3 次 => 开始(1次翻转) + 结束(2次翻转) => inMath=0
          inMath = 0
          ensureEmptyLineAfter()
        }
      }
    }
  }
}
' "$file" > "$tmpfile"

# 如果脚本能运行到这里，说明 $$ 已匹配，且插空行操作已完成
mv "$tmpfile" "$file"
