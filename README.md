# Literature Agent

一个基于 LLM 的文献处理系统，能够对学术 PDF 进行结构化解析、元数据提取、章节构建和多层级总结。

## 代码框架

```
.
├── main.py                          # CLI 入口脚本
├── pyproject.toml                   # 项目配置
├── pdf_parser/                      # PDF 解析模块（独立子模块）
│
└── literature_agent/                # 核心代码
    ├── pipeline.py                  # 流程编排 + checkpoint 钩子
    ├── storage.py                   # JSON 读写（内置 storage 实现）
    ├── models/
    │   └── document.py              # 核心数据模型 Document
    ├── agents/
    │   ├── base.py                  # BaseAgent 抽象基类
    │   ├── metadata.py              # MetadataAgent — 提取文献元数据
    │   ├── structure.py             # StructureAgent — 构建章节树
    │   └── summary.py               # SummaryAgent — 层次化总结
    └── utils/
        ├── llm_client.py            # LLM 客户端（配置 API）
        └── pdf_reader.py            # PDF 解析封装（依赖 pdf_parser）
```

### Pipeline 流程

```
PDF bytes → PDF 解析 (content + blocks) → 元数据提取 (metadata)
     → 章节结构构建 (structure) → 层次化总结 (summaries)
     → Document 对象
```

Pipeline 在每个步骤完成后触发 checkpoint 钩子。`ProcessingState` 记录完成状态，调用方通过 `document` 参数传入已有状态，Pipeline 自动跳过已完成步骤。

### 核心数据模型

`Document` 是系统的核心数据结构，包含以下字段：

| 字段 | 类型 | 说明 | 由谁填充 |
|------|------|------|----------|
| `id` | `str` | PDF 内容 SHA256 | 解析时生成 |
| `content` | `DocumentContent` | 全文文本和分页 | `pdf_reader` |
| `blocks` | `dict[int, DocumentBlock]` | 文本块字典（id → block） | `pdf_reader` |
| `metadata` | `DocumentMetadata` | 标题/作者/年份等 | `MetadataAgent` |
| `structure` | `DocumentStructure` | 章节树 | `StructureAgent` |
| `summaries` | `DocumentSummary` | 层次化摘要 | `SummaryAgent` |
| `processing` | `ProcessingState` | 处理状态 | pipeline |

### Agent 说明

| Agent | 输入 | 输出 | 方式 |
|-------|------|------|------|
| **MetadataAgent** | 前 2 页文本 | `DocumentMetadata` | 单次 LLM 调用，`response_format` 结构化提取 |
| **StructureAgent** | `Document.blocks` | `DocumentStructure` | 两步：LLM 标题检测 → Python 栈构建章节树 |
| **SummaryAgent** | `SectionNode` 树 | `DocumentSummary` | 自底向上后序遍历，逐层调用 LLM |

## 快速开始

### 安装依赖

```bash
uv sync
```

### 配置 API Key

创建 `.env` 文件：

```bash
CHATECNU_API_KEY=your_api_key_here
```

### 运行 Pipeline

```bash
# 默认输出（与 PDF 同目录）
uv run main.py path/to/paper.pdf

# 指定输出路径
uv run main.py path/to/paper.pdf -o output/result.json
```

## 在代码中调用

```python
from pathlib import Path
from literature_agent.pipeline import run
from literature_agent.storage import save as save_json, load as load_json

# 首次处理
pdf_bytes = Path("paper.pdf").read_bytes()
doc = run(pdf_bytes, checkpoint=lambda d: print(f"Step done: {d.id}"))

# 从 checkpoint 恢复
saved = load_json("paper.pdf", "paper.document.json")
doc = run(pdf_bytes, document=saved, checkpoint=save_json)

# 重新生成摘要（将 summary_done 重置后传入）
saved.processing.summary_done = False
doc = run(pdf_bytes, document=saved, checkpoint=save_json)

# 访问处理结果
print(doc.metadata.title)                   # 标题
print(doc.metadata.authors)                 # 作者列表
print(f"章节数: {len(doc.structure.nodes)}") # 章节树节点数
print(f"一句话摘要: {doc.summaries.one_sentence_summary}")
```

## Pipeline API

```python
def run(
    pdf_bytes: bytes,
    document: Document | None = None,
    *,
    checkpoint: Callable[[Document], None] = lambda _: None,
) -> Document
```

| 参数 | 说明 |
|------|------|
| `pdf_bytes` | PDF 原始字节 |
| `document` | 已有 Document 状态，用于断点续跑。None 表示全新处理 |
| `checkpoint` | 每步完成后调用的回调，默认 no-op。用于持久化当前进度 |
