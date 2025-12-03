# Train BPE 笔记（cs336_basics/train_bpe.py）

## 背景
- 目标：复现 GPT‑2 风格的 BPE 训练，生成与参考 merges/vocab 一致的结果。
- 关键：分词模式 `GPT2_SPLIT_PATTERN`，特殊符号（如 `<|endoftext|>`）需要跳过合并。

## 遇到的问题
- 测试 `tests/test_train_bpe.py` 不通过， merges 与参考在第 31 次合并开始偏离。
- 表现：同频率的 pair 选择顺序与参考实现相反，后续所有 merges/vocab 全部不同。

## 根因分析
- 堆元素形如 `(-freq, pair)`，频率相同时 `heapq` 默认按元组字典序升序比较 `pair`。
- 参考实现（GPT‑2）在同频时会选择**字典序更大的 pair**，与默认升序相反。
- 例子：频率相同的 `(b' ', b'd')` 与 `(b' a', b'nd')`。默认堆先取前者，参考先取后者，导致后续合并路径完全不同。

## 修复方案
- 自定义比较包装 `_RevPair`，在 `__lt__` 中返回 `self.pair > other.pair`，将同频 tie‑break 反转为字典序降序。
- 堆元素改为 `(-freq, _RevPair(pair))`，频率仍然用负号取最大，同频时按降序 pair 先出堆。

### 关键代码片段（精简版）
```python
class _RevPair:
    __slots__ = ("pair",)
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair
    def __lt__(self, other: "_RevPair") -> bool:
        return self.pair > other.pair  # 反转字典序

heap = [(-freq, _RevPair(pair)) for pair, freq in stats.items()]
heapq.heapify(heap)
```

## 训练算法流程（当前实现）
1) 读取文本，按特殊 token 切分，跳过特殊 token 本身。
2) 用 GPT‑2 正则拆分词，再转为字节 tuple；统计 `word_counts`。
3) 初始化 vocab（0–255）+ special tokens。
4) 构建 `stats`（pair 频次）与 `indices`（pair 出现的单词索引），推入堆。
5) 循环直到 vocab_size：
   - 弹出最高频 pair（同频按 `_RevPair` 降序）。
   - 生成新符号、记录 merges。
   - 更新受影响单词并维护 `stats`、`indices`，需要懒惰删除和增量更新。
6) 返回 `vocab, merges`。

## 常见坑
- 同频 tie‑break：如果不用自定义比较，结果将与参考偏离。
- 特殊 token 处理：必须在分块时跳过其合并，且仍要写入 vocab。
- 懒惰删除：堆顶可能是过期频次，需要对比 `stats[pair]`。
- 统计更新：左右邻居的计数要同步增减，否则频率会漂移。

## 验证
- 在脚本中比对参考 merges：`all match 243`（与 `tests/fixtures/train-bpe-reference-merges.txt` 一致）。
- 预计 `tests/test_train_bpe.py` 通过（需安装 `torch` 以加载 conftest）。

## 参考命令
```bash
pytest tests/test_train_bpe.py
```

## 结论
- 核心修复是同频率下的排序规则；通过 `_RevPair` 实现字典序降序，保证 merges 与参考一致。***
