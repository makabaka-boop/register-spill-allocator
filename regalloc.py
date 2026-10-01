#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regalloc.py —— 不解析源语言的寄存器分配器。

输入是抽象的控制流图（CFG）：每条指令只声明“先读哪些变量、再定义哪个变量”，
不涉及任何源语言语法。分配流程：

  1. 活跃变量分析：在（可能含回边的）控制流图上迭代求最小不动点；
  2. 构建干涉图：
     a. 同一程序点（每条指令之前、块末尾）同时活跃的变量两两干涉；
     b. 每条指令的 def 与该指令之后仍活跃（live-after）的每个变量干涉
        （写回寄存器会破坏这些活跃值，即使 def 本身随即死亡）；
  3. 枚举溢出集合：找出总溢出代价最小、且其余变量能用各自可用寄存器
     合法着色的集合；代价并列时取排序后溢出变量 ID 序列最小者；
  4. 在选定的溢出集合下，取字典序最小的寄存器映射
     （按变量 ID 升序排列各变量分到的寄存器名，比较字典序）。

输入格式（逐行；'#' 到行尾为注释；空行忽略）：

    var <name> <cost> <reg> <reg> [<reg>] [<reg>]   # 2~4 个可用寄存器，cost 为正整数
    block <name>                                    # 开始一个基本块（第一个声明的块为入口）
    succ <name> [<name> ...]                        # 后继块（可省略，省略表示无后继）
    instr [<use> ...] -> <def | ->                  # 先读 use 列表，再定义 def；'-' 表示无定义

所有 var 行必须先于第一个 block 行；块内至多一条 succ 行、任意多条 instr 行。

出现以下任一情况整份拒绝（退出码 1，不输出任何分配结果）：
  - 指令引用未声明的变量；
  - succ 引用未声明的块；
  - 重复的块名（以及重复的变量名）；
  - 其他格式错误：代价非正整数、可用寄存器个数不在 2~4、寄存器重复、
    指令缺少 '->'、'->' 后不是恰好一个 def、var 出现在 block 之后、
    变量超过 12 个、块超过 30 个、没有任何块等。

用法： python3 regalloc.py [输入文件]     （缺省从标准输入读取）
"""

import re
import sys

MAX_VARS = 12
MAX_BLOCKS = 30
MIN_REGS = 2
MAX_REGS = 4

_POSINT = re.compile(r"^[0-9]+$")


class InputError(Exception):
    """输入非法：整份拒绝。"""


def _is_ident(tok):
    return (
        bool(tok)
        and (tok[0].isalpha() or tok[0] == "_")
        and all(ch.isalnum() or ch == "_" for ch in tok)
    )


class Instr:
    __slots__ = ("uses", "def_")

    def __init__(self, uses, def_):
        self.uses = tuple(uses)  # 先读取的变量（按声明顺序）
        self.def_ = def_         # 再定义的变量名；None 表示本条指令无定义

    def __repr__(self):
        return f"Instr(uses={self.uses!r}, def_={self.def_!r})"


class Block:
    __slots__ = ("name", "succs", "instrs", "has_succ_line")

    def __init__(self, name):
        self.name = name
        self.succs = []          # 后继块名（保持声明顺序）
        self.instrs = []
        self.has_succ_line = False


class Program:
    def __init__(self):
        self.var_cost = {}       # 变量名 -> 溢出代价（正整数）
        self.var_regs = {}       # 变量名 -> 可用寄存器元组（2~4 个）
        self.blocks = {}         # 块名 -> Block
        self.block_order = []    # 块声明顺序（第一个为入口块）

    @property
    def variables(self):
        return sorted(self.var_cost)


# ---------------------------------------------------------------- 输入解析

def parse(text):
    prog = Program()
    current = None
    seen_block = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        head = toks[0]
        try:
            if head == "var":
                if seen_block:
                    raise InputError("'var' 必须出现在第一个 'block' 之前")
                if not 3 + MIN_REGS <= len(toks) <= 3 + MAX_REGS:
                    raise InputError(
                        f"格式应为 var <name> <cost> <reg>...（{MIN_REGS}~{MAX_REGS} 个寄存器）")
                _, name, cost_s, *regs = toks
                if not _is_ident(name):
                    raise InputError(f"非法变量名 {name!r}")
                if name in prog.var_cost:
                    raise InputError(f"重复声明的变量 {name!r}")
                if not _POSINT.match(cost_s) or int(cost_s) == 0:
                    raise InputError(f"变量 {name!r} 的溢出代价必须为正整数")
                if len(set(regs)) != len(regs):
                    raise InputError(f"变量 {name!r} 的可用寄存器有重复")
                for r in regs:
                    if not _is_ident(r):
                        raise InputError(f"非法寄存器名 {r!r}")
                prog.var_cost[name] = int(cost_s)
                prog.var_regs[name] = tuple(regs)
            elif head == "block":
                seen_block = True
                if len(toks) != 2:
                    raise InputError("格式应为 block <name>")
                name = toks[1]
                if not _is_ident(name):
                    raise InputError(f"非法块名 {name!r}")
                if name in prog.blocks:
                    raise InputError(f"重复声明的基本块 {name!r}")
                current = Block(name)
                prog.blocks[name] = current
                prog.block_order.append(name)
            elif head == "succ":
                if current is None:
                    raise InputError("'succ' 出现在任何 'block' 之外")
                if current.has_succ_line:
                    raise InputError(f"块 {current.name!r} 有重复的 'succ' 行")
                current.has_succ_line = True
                seen = set()
                for s in toks[1:]:
                    if not _is_ident(s):
                        raise InputError(f"非法后继块名 {s!r}")
                    if s in seen:
                        raise InputError(f"块 {current.name!r} 的后继 {s!r} 重复")
                    seen.add(s)
                    current.succs.append(s)
            elif head == "instr":
                if current is None:
                    raise InputError("'instr' 出现在任何 'block' 之外")
                if toks.count("->") != 1:
                    raise InputError("'instr' 需要恰好一个 '->'")
                arrow = toks.index("->")
                uses = toks[1:arrow]
                defs = toks[arrow + 1:]
                if len(defs) != 1:
                    raise InputError("'->' 之后必须恰好是一个定义变量或 '-'")
                for u in uses:
                    if u not in prog.var_cost:
                        raise InputError(f"指令读取了未声明的变量 {u!r}")
                d = defs[0]
                if d == "-":
                    d = None
                elif d not in prog.var_cost:
                    raise InputError(f"指令定义了未声明的变量 {d!r}")
                current.instrs.append(Instr(uses, d))
            else:
                raise InputError(f"无法识别的指令 {head!r}")
        except InputError as exc:
            raise InputError(f"第 {lineno} 行: {exc}") from None
    if not prog.block_order:
        raise InputError("没有任何基本块")
    if len(prog.block_order) > MAX_BLOCKS:
        raise InputError(f"基本块数量超过上限 {MAX_BLOCKS}")
    if len(prog.var_cost) > MAX_VARS:
        raise InputError(f"变量数量超过上限 {MAX_VARS}")
    for b in prog.block_order:
        for s in prog.blocks[b].succs:
            if s not in prog.blocks:
                raise InputError(f"块 {b!r} 的后继 {s!r} 未声明")
    return prog


# ---------------------------------------------------------------- 活跃变量分析

def compute_liveness(prog):
    """返回 (use, defv, live_in, live_out)。

    use[B]  = B 中先于任何定义被读取的变量（向上暴露使用）
    defv[B] = B 中定义过的变量
    liveOut[B] = ∪ liveIn[S]（S 为 B 的后继）
    liveIn[B]  = use[B] ∪ (liveOut[B] − defv[B])
    在含回边的控制流图上迭代至最小不动点。
    """
    use, defv = {}, {}
    for b in prog.block_order:
        defined = set()
        u, d = set(), set()
        for ins in prog.blocks[b].instrs:
            for x in ins.uses:
                if x not in defined:
                    u.add(x)
            if ins.def_ is not None:
                defined.add(ins.def_)
                d.add(ins.def_)
        use[b], defv[b] = u, d
    live_in = {b: set() for b in prog.block_order}
    live_out = {b: set() for b in prog.block_order}
    changed = True
    while changed:
        changed = False
        for b in reversed(prog.block_order):
            out = set()
            for s in prog.blocks[b].succs:
                out |= live_in[s]
            inn = use[b] | (out - defv[b])
            if inn != live_in[b] or out != live_out[b]:
                live_in[b], live_out[b] = inn, out
                changed = True
    return use, defv, live_in, live_out


# ---------------------------------------------------------------- 干涉图

def compute_interference(prog, live_out):
    """构建干涉图（无向，邻接集）。规则：
    1. 同一程序点（每条指令之前、块末尾）同时活跃的变量两两干涉；
    2. 指令定义的变量与该指令之后仍活跃的每个变量干涉。
    """
    adj = {v: set() for v in prog.var_cost}

    def add_edge(x, y):
        if x != y:
            adj[x].add(y)
            adj[y].add(x)

    def add_clique(members):
        ms = list(members)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                add_edge(ms[i], ms[j])

    for b in prog.block_order:
        live = set(live_out[b])
        add_clique(live)  # 块末尾程序点
        for ins in reversed(prog.blocks[b].instrs):
            if ins.def_ is not None:
                for other in live:
                    add_edge(ins.def_, other)  # 写回时这些值仍活跃
                live.discard(ins.def_)
            live |= set(ins.uses)
            add_clique(live)  # 指令之前的程序点
    return adj


# ---------------------------------------------------------------- 溢出与着色

def solve(prog, adj):
    """返回 (溢出变量 ID 升序列表, 寄存器映射 dict, 总溢出代价)。

    优化目标依次：
      1. 总溢出代价最小；
      2. 并列时排序后的溢出变量 ID 序列最小（字典序）；
      3. 寄存器映射字典序最小（按变量 ID 升序比较各变量分到的寄存器名）。
    """
    variables = prog.variables
    n = len(variables)

    def colorable(kept):
        kept_set = set(kept)
        order = sorted(kept, key=lambda v: -len(adj[v] & kept_set))
        assign = {}

        def bt(i):
            if i == len(order):
                return True
            v = order[i]
            used = {assign[u] for u in adj[v] if u in assign}
            for r in prog.var_regs[v]:
                if r not in used:
                    assign[v] = r
                    if bt(i + 1):
                        return True
                    del assign[v]
            return False

        return bt(0)

    best_cost, best_spill = None, None
    for mask in range(1 << n):
        spill = [variables[i] for i in range(n) if mask & (1 << i)]
        cost = sum(prog.var_cost[v] for v in spill)
        if best_cost is not None and (cost > best_cost
                                      or (cost == best_cost and spill >= best_spill)):
            continue
        kept = [variables[i] for i in range(n) if not mask & (1 << i)]
        if colorable(kept):
            best_cost, best_spill = cost, spill
            if cost == 0:  # 代价为正整数，0 已是最优
                break
    kept = [v for v in variables if v not in best_spill]
    coloring = _lexmin_coloring(kept, adj, prog.var_regs)
    return best_spill, coloring, best_cost


def _lexmin_coloring(kept_sorted, adj, var_regs):
    """按变量 ID 升序依次尝试升序寄存器，第一组可行解即字典序最小映射。"""
    assign = {}

    def bt(i):
        if i == len(kept_sorted):
            return True
        v = kept_sorted[i]
        used = {assign[u] for u in adj[v] if u in assign}
        for r in sorted(var_regs[v]):
            if r not in used:
                assign[v] = r
                if bt(i + 1):
                    return True
                del assign[v]
        return False

    bt(0)
    return assign


# ---------------------------------------------------------------- 输出

def _fmt_set(s):
    return " ".join(sorted(s)) if s else "(empty)"


def format_report(prog, use, defv, live_in, live_out, adj, spill, coloring, cost):
    out = ["=== Liveness ==="]
    for b in prog.block_order:
        out.append(f"block {b}")
        out.append(f"  use    : {_fmt_set(use[b])}")
        out.append(f"  def    : {_fmt_set(defv[b])}")
        out.append(f"  liveIn : {_fmt_set(live_in[b])}")
        out.append(f"  liveOut: {_fmt_set(live_out[b])}")
    out.append("=== Interference ===")
    edges = sorted({(a, c) for a in adj for c in adj[a] if a < c})
    if edges:
        out.extend(f"{a} -- {c}" for a, c in edges)
    else:
        out.append("(none)")
    out.append("=== Spill ===")
    out.append(f"cost {cost}: " + (" ".join(spill) if spill else "(none)"))
    out.append("=== Allocation ===")
    for v in prog.variables:
        out.append(f"{v} = {coloring[v]}" if v in coloring else f"{v} = SPILL")
    return "\n".join(out)


def main(argv):
    if len(argv) > 2:
        print("usage: regalloc.py [input-file]", file=sys.stderr)
        return 2
    try:
        if len(argv) == 2:
            with open(argv[1], "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = sys.stdin.read()
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        prog = parse(text)
    except InputError as exc:
        print(f"input rejected: {exc}", file=sys.stderr)
        return 1
    use, defv, live_in, live_out = compute_liveness(prog)
    adj = compute_interference(prog, live_out)
    spill, coloring, cost = solve(prog, adj)
    print(format_report(prog, use, defv, live_in, live_out, adj, spill, coloring, cost))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
