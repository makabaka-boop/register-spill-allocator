# -*- coding: utf-8 -*-
"""regalloc 的测试：

1. 手算用例：回边（循环/自循环/嵌套循环）、分支汇合（含汇合点的保守活跃性）、
   不可达块、死定义干涉、两级 tie-break；
2. 非法输入整份拒绝；
3. 对拍：随机小图上枚举全部溢出集合与全部着色（itertools.product 暴力），
   与分配器结果逐一比对。
"""

import itertools
import random
import unittest

from regalloc import (
    Block,
    Instr,
    InputError,
    Program,
    compute_interference,
    compute_liveness,
    parse,
    solve,
)


def analyze(text):
    prog = parse(text)
    use, defv, live_in, live_out = compute_liveness(prog)
    adj = compute_interference(prog, live_out)
    spill, coloring, cost = solve(prog, adj)
    return prog, use, defv, live_in, live_out, adj, spill, coloring, cost


def edge_set(adj):
    return {frozenset((a, b)) for a in adj for b in adj[a] if a != b}


def fs(*pairs):
    return {frozenset(p) for p in pairs}


# ---------------------------------------------------------------- 手算用例

class TestLoopBackEdge(unittest.TestCase):
    """回边：i 是循环携带变量，one 是循环不变量，n 定义后不再使用。"""

    PROG = """
        var i 3 r1 r2
        var n 4 r1 r2
        var one 2 r1 r2
        block entry
        succ header
        instr -> one
        instr -> i
        instr -> n
        block header
        succ body exit
        block body
        succ header
        instr i one -> i
        block exit
        instr i -> -
    """

    def test_liveness(self):
        _, _, _, live_in, live_out, *_ = analyze(self.PROG)
        self.assertEqual(live_in["entry"], set())
        self.assertEqual(live_out["entry"], {"i", "one"})
        self.assertEqual(live_in["header"], {"i", "one"})
        self.assertEqual(live_out["header"], {"i", "one"})
        self.assertEqual(live_in["body"], {"i", "one"})
        self.assertEqual(live_out["body"], {"i", "one"})
        self.assertEqual(live_in["exit"], {"i"})
        self.assertEqual(live_out["exit"], set())

    def test_interference_and_alloc(self):
        _, _, _, _, _, adj, spill, coloring, cost = analyze(self.PROG)
        # n 虽定义后即死，但其写回时 i、one 仍活跃 -> 三者构成三角形
        self.assertEqual(edge_set(adj), fs(("i", "n"), ("i", "one"), ("n", "one")))
        self.assertEqual(spill, ["one"])   # 代价最小者被溢出
        self.assertEqual(cost, 2)
        self.assertEqual(coloring, {"i": "r1", "n": "r2"})


class TestSelfLoop(unittest.TestCase):
    PROG = """
        var x 1 r1 r2
        var y 1 r1 r2
        block loop
        succ loop out
        instr x -> y
        block out
        instr y -> -
    """

    def test_all(self):
        _, _, _, live_in, live_out, adj, spill, coloring, _ = analyze(self.PROG)
        self.assertEqual(live_in["loop"], {"x"})
        self.assertEqual(live_out["loop"], {"x", "y"})
        self.assertEqual(live_in["out"], {"y"})
        self.assertEqual(edge_set(adj), fs(("x", "y")))
        self.assertEqual(spill, [])
        self.assertEqual(coloring, {"x": "r1", "y": "r2"})


class TestNestedLoops(unittest.TestCase):
    """嵌套循环 + 两条回边（inner 自循环、inner->outer）。"""

    PROG = """
        var x 1 r1 r2
        var y 1 r1 r2
        var z 1 r1 r2
        block entry
        succ outer
        instr -> x
        block outer
        succ inner done
        instr x -> y
        block inner
        succ inner outer
        instr y -> z
        instr z -> y
        block done
        instr y -> -
    """

    def test_all(self):
        _, _, _, live_in, live_out, adj, spill, coloring, _ = analyze(self.PROG)
        self.assertEqual(live_in["entry"], set())
        self.assertEqual(live_out["entry"], {"x"})
        self.assertEqual(live_in["outer"], {"x"})
        self.assertEqual(live_out["outer"], {"x", "y"})
        self.assertEqual(live_in["inner"], {"x", "y"})
        self.assertEqual(live_out["inner"], {"x", "y"})
        self.assertEqual(live_in["done"], {"y"})
        # y 与 z 不同时活跃（z 的最后用途定义 y），故无 y-z 边
        self.assertEqual(edge_set(adj), fs(("x", "y"), ("x", "z")))
        self.assertEqual(spill, [])
        self.assertEqual(coloring, {"x": "r1", "y": "r2", "z": "r2"})


class TestBranchMerge(unittest.TestCase):
    """分支汇合：b 不在 left 中使用，但仍活跃地穿过 left。"""

    PROG = """
        var a 1 r1 r2
        var b 1 r1 r2
        block entry
        succ left right
        instr -> a
        instr -> b
        block left
        succ join
        instr a -> -
        block right
        succ join
        instr b -> -
        block join
        instr a b -> -
    """

    def test_all(self):
        _, _, _, live_in, live_out, adj, spill, coloring, _ = analyze(self.PROG)
        self.assertEqual(live_in["join"], {"a", "b"})
        self.assertEqual(live_out["left"], {"a", "b"})
        self.assertEqual(live_in["left"], {"a", "b"})
        self.assertEqual(live_out["right"], {"a", "b"})
        self.assertEqual(live_out["entry"], {"a", "b"})
        self.assertEqual(live_in["entry"], set())
        self.assertEqual(edge_set(adj), fs(("a", "b")))
        self.assertEqual(spill, [])
        self.assertEqual(coloring, {"a": "r1", "b": "r2"})


class TestMergeConservative(unittest.TestCase):
    """汇合点的保守活跃性：c、d 都被 join 使用，于是各自活跃地穿过另一条分支，
    在 entry 出口形成 K4，必须溢出两个变量。"""

    PROG = """
        var a 1 r1 r2
        var b 1 r1 r2
        var c 1 r1 r2
        var d 1 r1 r2
        var e 1 r1 r2
        block entry
        succ left right
        instr -> a
        instr -> b
        block left
        succ join
        instr a -> c
        block right
        succ join
        instr b -> d
        block join
        instr c d -> e
    """

    def test_all(self):
        _, _, _, live_in, live_out, adj, spill, coloring, cost = analyze(self.PROG)
        self.assertEqual(live_in["join"], {"c", "d"})
        self.assertEqual(live_out["left"], {"c", "d"})
        self.assertEqual(live_in["left"], {"a", "d"})
        self.assertEqual(live_in["right"], {"b", "c"})
        self.assertEqual(live_out["entry"], {"a", "b", "c", "d"})
        self.assertEqual(live_in["entry"], {"c", "d"})
        k4 = fs(("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d"))
        self.assertEqual(edge_set(adj), k4)
        # K4 用 2 个寄存器不可着色，至少溢出 2 个；等代价时取 ID 序列最小者
        self.assertEqual(spill, ["a", "b"])
        self.assertEqual(cost, 2)
        self.assertEqual(coloring, {"c": "r1", "d": "r2", "e": "r1"})


class TestUnreachableBlock(unittest.TestCase):
    PROG = """
        var a 1 r1 r2
        block entry
        succ
        instr -> a
        block dead
        succ
        instr a -> a
    """

    def test_all(self):
        _, _, _, live_in, live_out, adj, spill, coloring, _ = analyze(self.PROG)
        self.assertEqual(live_in["dead"], {"a"})   # 不可达块同样参与不动点
        self.assertEqual(live_out["dead"], set())
        self.assertEqual(edge_set(adj), set())
        self.assertEqual(spill, [])
        self.assertEqual(coloring, {"a": "r1"})


class TestSpillTieBreakIdSequence(unittest.TestCase):
    """p、q 溢出代价相同且都为最小，取排序后 ID 序列最小的 [p]。"""

    PROG = """
        var p 5 r1 r2
        var q 5 r1 r2
        var r 9 r1 r2
        block blk
        instr -> p
        instr -> q
        instr -> r
        instr p q r -> -
    """

    def test_all(self):
        _, _, _, _, _, adj, spill, coloring, cost = analyze(self.PROG)
        self.assertEqual(edge_set(adj), fs(("p", "q"), ("p", "r"), ("q", "r")))
        self.assertEqual(spill, ["p"])
        self.assertEqual(cost, 5)
        self.assertEqual(coloring, {"q": "r1", "r": "r2"})


class TestMinCostSpill(unittest.TestCase):
    """K4 + 2 个寄存器：必须溢出 2 个，选总代价最小的组合而非 ID 最小的组合。"""

    PROG = """
        var v1 4 r1 r2
        var v2 3 r1 r2
        var v3 2 r1 r2
        var v4 1 r1 r2
        block blk
        instr -> v1
        instr -> v2
        instr -> v3
        instr -> v4
        instr v1 v2 v3 v4 -> -
    """

    def test_all(self):
        _, _, _, _, _, adj, spill, coloring, cost = analyze(self.PROG)
        self.assertEqual(len(edge_set(adj)), 6)  # K4
        self.assertEqual(spill, ["v3", "v4"])    # 1 + 2 = 3 最小
        self.assertEqual(cost, 3)
        self.assertEqual(coloring, {"v1": "r1", "v2": "r2"})


class TestLexMinMapping(unittest.TestCase):
    """可用寄存器声明顺序不影响字典序：映射按寄存器名升序取。"""

    PROG = """
        var a 1 r2 r1
        var b 1 r1 r2
        block blk
        instr -> a
        instr -> b
        instr a b -> -
    """

    def test_all(self):
        *_, coloring, _ = analyze(self.PROG)
        self.assertEqual(coloring, {"a": "r1", "b": "r2"})


class TestZeroVariables(unittest.TestCase):
    def test_all(self):
        *_, spill, coloring, cost = analyze("block only\ninstr -> -\n")
        self.assertEqual(spill, [])
        self.assertEqual(coloring, {})
        self.assertEqual(cost, 0)


# ---------------------------------------------------------------- 非法输入整份拒绝

class TestRejections(unittest.TestCase):
    CASES = {
        "unknown use var": "var a 1 r1 r2\nblock b\ninstr z -> a\n",
        "unknown def var": "var a 1 r1 r2\nblock b\ninstr a -> z\n",
        "unknown successor": "var a 1 r1 r2\nblock b\nsucc nowhere\n",
        "duplicate block": "var a 1 r1 r2\nblock b\nblock b\n",
        "duplicate var": "var a 1 r1 r2\nvar a 2 r1 r2\nblock b\n",
        "zero cost": "var a 0 r1 r2\nblock b\n",
        "non-numeric cost": "var a x r1 r2\nblock b\n",
        "too few regs": "var a 1 r1\nblock b\n",
        "too many regs": "var a 1 r1 r2 r3 r4 r5\nblock b\n",
        "duplicate reg": "var a 1 r1 r1\nblock b\n",
        "instr outside block": "var a 1 r1 r2\ninstr -> a\n",
        "succ outside block": "var a 1 r1 r2\nsucc b\n",
        "var after block": "var a 1 r1 r2\nblock b\nvar c 1 r1 r2\n",
        "missing arrow": "var a 1 r1 r2\nblock b\ninstr a\n",
        "two defs": "var a 1 r1 r2\nvar b 1 r1 r2\nblock blk\ninstr -> a b\n",
        "empty input": "",
        "no blocks": "var a 1 r1 r2\n",
        "too many vars": "".join(f"var v{i} 1 r1 r2\n" for i in range(13)) + "block b\n",
        "too many blocks": "var a 1 r1 r2\n" + "".join(f"block b{i}\n" for i in range(31)),
        "duplicate successor": "var a 1 r1 r2\nblock b\nsucc b b\n",
        "second succ line": "var a 1 r1 r2\nblock b\nsucc c\nsucc c\nblock c\n",
        "bad ident": "var 1a 1 r1 r2\nblock b\n",
    }

    def test_all_rejected(self):
        for name, text in self.CASES.items():
            with self.subTest(case=name):
                with self.assertRaises(InputError):
                    parse(text)


# ---------------------------------------------------------------- 对拍：暴力枚举

def brute_force(prog, adj):
    """独立实现：枚举全部溢出子集；对每个子集用 itertools.product 枚举
    全部寄存器赋值检验可行性。返回 (cost, spill_list, lexmin_mapping)。"""
    variables = sorted(prog.var_cost)
    edges = [(a, b) for a in variables for b in adj[a] if a < b]
    best = None
    n = len(variables)
    for mask in range(1 << n):
        spill = [variables[i] for i in range(n) if mask & (1 << i)]
        cost = sum(prog.var_cost[v] for v in spill)
        if best is not None and (cost, spill) >= (best[0], best[1]):
            continue
        kept = [v for i, v in enumerate(variables) if not mask & (1 << i)]
        mapping = None
        for combo in itertools.product(*(sorted(prog.var_regs[v]) for v in kept)):
            m = dict(zip(kept, combo))
            if all(m[a] != m[b] for a, b in edges if a in m and b in m):
                mapping = m
                break
        if mapping is None:
            continue
        best = (cost, spill, mapping)
    return best


def random_program(rng):
    """随机小图：任意后继（含自环与回边）、任意读写序列。"""
    prog = Program()
    nvars = rng.randint(1, 5)
    pool = [f"r{i}" for i in range(rng.randint(2, 3))]
    for i in range(nvars):
        v = f"v{i}"
        prog.var_cost[v] = rng.randint(1, 9)
        prog.var_regs[v] = tuple(sorted(rng.sample(pool, rng.randint(2, len(pool)))))
    names = [f"b{i}" for i in range(rng.randint(1, 4))]
    for name in names:
        prog.blocks[name] = Block(name)
        prog.block_order.append(name)
    for name in names:
        blk = prog.blocks[name]
        blk.succs = rng.sample(names, rng.randint(0, min(2, len(names))))
        for _ in range(rng.randint(0, 4)):
            uses = rng.sample(sorted(prog.var_cost), rng.randint(0, min(2, nvars)))
            d = rng.choice(sorted(prog.var_cost) + [None])
            blk.instrs.append(Instr(uses, d))
    return prog


def render(prog):
    lines = []
    for v in sorted(prog.var_cost):
        lines.append("var {} {} {}".format(v, prog.var_cost[v], " ".join(prog.var_regs[v])))
    for b in prog.block_order:
        lines.append(f"block {b}")
        blk = prog.blocks[b]
        lines.append("succ" + "".join(f" {s}" for s in blk.succs))
        for ins in blk.instrs:
            lines.append("instr {} -> {}".format(" ".join(ins.uses), ins.def_ or "-"))
    return "\n".join(lines) + "\n"


class TestCrossCheck(unittest.TestCase):
    def test_random_against_brute_force(self):
        rng = random.Random(20261001)
        for trial in range(400):
            prog = random_program(rng)
            text = render(prog)

            # 解析器往返一致
            prog2 = parse(text)
            self.assertEqual(prog.var_cost, prog2.var_cost)
            self.assertEqual(prog.var_regs, prog2.var_regs)
            self.assertEqual(prog.block_order, prog2.block_order)
            for b in prog.block_order:
                self.assertEqual(prog.blocks[b].succs, prog2.blocks[b].succs)
                self.assertEqual(
                    [(i.uses, i.def_) for i in prog.blocks[b].instrs],
                    [(i.uses, i.def_) for i in prog2.blocks[b].instrs],
                )

            use, defv, live_in, live_out = compute_liveness(prog)
            # 活跃方程自洽（不动点性质）
            for b in prog.block_order:
                expect_out = set().union(*(live_in[s] for s in prog.blocks[b].succs))
                self.assertEqual(live_out[b], expect_out, msg=text)
                self.assertEqual(live_in[b], use[b] | (live_out[b] - defv[b]), msg=text)

            adj = compute_interference(prog, live_out)
            # 干涉图无自环且对称
            for v in adj:
                self.assertNotIn(v, adj[v])
                for u in adj[v]:
                    self.assertIn(v, adj[u])

            spill, coloring, cost = solve(prog, adj)
            bf = brute_force(prog, adj)
            self.assertIsNotNone(bf, msg=text)  # 全部溢出必然可行
            self.assertEqual((cost, spill), (bf[0], bf[1]), msg=text)
            self.assertEqual(coloring, bf[2], msg=text)

            # 着色合法性：恰好覆盖未溢出变量、寄存器在可用集内、干涉边两端不同色
            self.assertEqual(set(coloring), set(prog.var_cost) - set(spill))
            for v, r in coloring.items():
                self.assertIn(r, prog.var_regs[v])
            for a in coloring:
                for c in adj[a]:
                    if c in coloring:
                        self.assertNotEqual(coloring[a], coloring[c], msg=text)


if __name__ == "__main__":
    unittest.main()
