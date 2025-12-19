# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

# CS336_lab1
# assignment1-basics

## Resource Accounting (GPT-2 XL)

Based on the configuration:
- `vocab_size`: 50,257
- `context_length` (L): 1,024
- `num_layers`: 48
- `d_model` (D): 1,600
- `num_heads` (H): 25
- `d_ff`: 6,400 (SwiGLU)

### FLOPs per Forward Pass (per Batch):

1. **Multi-Head Attention (MHA) per Layer**:
   - **QKV Projections**: $2 \times L \times D \times 3D = 15,728,640,000$ (15.73B)
   - **SDPA ($QK^T + AV$)**: $4 \times L^2 \times D = 6,710,886,400$ (6.71B)
   - **Output Projection**: $2 \times L \times D \times D = 5,242,880,000$ (5.24B)
   - **MHA Total**: **27,682,406,400** (27.68B)

2. **Feed-Forward Network (FFN) per Layer (SwiGLU)**:
   - **W1, W2, W3 Layers**: $3 \times (2 \times L \times D \times d_{ff}) = 62,914,560,000$ (62.91B)
   - **FFN Total**: **62,914,560,000** (62.91B)

3. **Total per Block**:
   - MHA + FFN = **90,596,966,400** (90.60B)

4. **Full Model (48 Blocks)**:
   - $48 \times 90.60B = 4,348,654,387,200$ (4,348.65B)

5. **LM Head (Final Linear Layer)**:
   - $2 \times L \times D \times \text{vocab\_size} = 164,682,137,600$ (164.68B)

### Grand Total:
**~4,513,336,524,800 FLOPs** (约 **4.51 TFLOPS**)

## Model Scaling Comparison (Small, Medium, Large)

Based on the same logic (T=1024, SwiGLU FFN), we compare the FLOPs distribution across different GPT-2 sizes:

| Component | GPT-2 Small (12L, 768D) | GPT-2 Medium (24L, 1024D) | GPT-2 Large (36L, 1280D) |
| :--- | :--- | :--- | :--- |
| **Total FLOPs** | **~349.6 GFLOPs** | **~1.03 TFLOPs** | **~2.26 TFLOPs** |
| **MHA (Attention)** | 27.6% | 29.9% | 30.0% |
| **FFN (Feed-Forward)** | 49.8% | 59.9% | 64.2% |
| **LM Head** | 22.6% | 10.2% | 5.8% |

### Key Findings:
- **FFN Dominance**: As models scale, the FFN blocks become the primary computational bottleneck, increasing from ~50% to nearly 65% of total FLOPs.
- **LM Head Dilution**: The "fixed cost" of the final linear layer (LM Head) is significantly diluted as the model grows deeper and wider, dropping from 22.6% to 5.8%.
- **Stable Attention**: Despite the $O(T^2)$ complexity, the Attention mechanism's relative contribution remains stable at around 30% for these model sizes and sequence lengths.

## 長上下文分析 (問題 e)

當我們將 GPT-2 XL 的 **上下文長度 ($T$)** 從 1,024 增加到 **16,384** (增加 16 倍) 時，計算資源的分佈發生了劇烈變化：

- **總 FLOPs**: 從約 4.51 TFLOPs 激增至 **約 149.5 TFLOPs** (增加約 33 倍)。
- **MHA 主導**: 由於點積注意力機制 (SDPA) 具有 $O(T^2)$ 的複雜度，多頭注意力 (MHA) 在總 FLOPs 中的佔比從 **約 30% 暴增至 約 66%**。
- **FFN 被稀釋**: 前饋神經網絡 (FFN) 與長度呈線性 $O(T)$ 關係，其佔比從 **約 62% 下降至 約 32%**。

### 結論：
在長上下文的情境下，標準 Attention 機制的二次方增長特性會壓倒所有其他模型組件，取代 FFN 成為 Transformer 架構的主要計算瓶頸。
