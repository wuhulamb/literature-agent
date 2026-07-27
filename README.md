# Literature Agent

一个基于 LLM 的文献处理系统，能够对学术 PDF 进行结构化解析、元数据提取、章节构建和多层级总结。

## 代码框架

```
.
├── main.py                          # 入口脚本
├── pyproject.toml                   # 项目配置
├── pdf_parser/                      # PDF 解析模块（独立子模块）
│
└── literature_agent/                # 核心代码
    ├── pipeline.py                  # 流程编排（串联各 Agent 执行）
    ├── models/
    │   └── document.py              # 核心数据模型 Document
    ├── agents/
    │   ├── base.py                  # BaseAgent 抽象基类
    │   ├── metadata.py              # MetadataAgent — 提取文献元数据
    │   ├── structure.py             # StructureAgent — 构建章节树
    │   └── summary.py               # SummaryAgent — 层次化总结
    └── utils/
        ├── llm_client.py            # LLM 客户端（配置 API）
        ├── pdf_reader.py            # PDF 解析封装（依赖 pdf_parser）
        └── json_storage.py          # Document 的 JSON 读写
```

### Pipeline 流程

```
PDF → PDF 解析 (content + blocks) → 元数据提取 (metadata)
     → 章节结构构建 (structure) → 层次化总结 (summaries)
     → 输出 JSON
```

每一步完成后保存 checkpoint，支持断点续跑（通过 `Document.processing` 状态判断）。

### 核心数据模型

`Document` 是系统的核心数据结构，包含以下字段：

| 字段 | 类型 | 说明 | 由谁填充 |
|------|------|------|----------|
| `id` | `str` | UUID | 解析时生成 |
| `path` | `str` | PDF 绝对路径 | 解析时生成 |
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

#### StructureAgent 细节

分为两步：

1. **HeadingDetector（LLM）**：判断每个 block 是否为章节标题，返回 `(block_id, is_heading, title, level)`
2. **StructureBuilder（Python）**：用栈维护当前章节层级，逐步构建 `SectionNode` 树

构建过程：

```
根节点（level=0）：文献标题（由 MetadataAgent 提供）
  │
  ├─ heading(level=1) → push（如 Introduction、Method）
  │    └─ heading(level=2) → push（子章节，如 2.1 Background）
  │         └─ block → 加入栈顶 section 的 block_ids
  │
  ├─ heading(level=1) → pop 到 level < 1，push（同级新章节）
  │    └─ block → 加入栈顶 section 的 block_ids
  │
  └─ ...
```

去重机制：通过 `existing_titles` 集合记录已存在的标题，LLM 误判的重复标题自动降级为 non-heading block。

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

## 在其他代码中调用

### 方式一：使用完整 Pipeline

```python
from literature_agent.pipeline import run

# 直接运行完整流程
run("paper.pdf")

# 指定输出路径
run("paper.pdf", output_path="output/paper.json")
```

### 方式二：解析 PDF 后获取 Document

```python
from literature_agent.utils.pdf_reader import parse

doc = parse("paper.pdf")
print(doc.id)
print(len(doc.blocks))   # blocks 为 dict[int, DocumentBlock]
```

### 方式三：单独使用某个 Agent

```python
from literature_agent.utils.pdf_reader import parse
from literature_agent.utils.llm_client import get_client
from literature_agent.agents.metadata import MetadataAgent

doc = parse("paper.pdf")

client = get_client()
agent = MetadataAgent(client)
doc = agent.run(doc)

print(doc.metadata.title)
print(doc.metadata.authors)
```

### 方式四：读写处理结果

```python
from literature_agent.utils import json_storage

# 读取已处理的 JSON
doc = json_storage.load("paper.pdf")
# 或指定路径
doc = json_storage.load("paper.pdf", output_path="output/paper.json")

# 保存
json_storage.save(doc, "paper.pdf")
# 或指定路径
json_storage.save(doc, output_path="output/result.json")

# 访问处理结果
print(doc.metadata.title)
print(doc.structure.nodes if doc.structure else None)
print(doc.summaries.document_summary)
```

### 方式五：作为 package 安装

```bash
# 在项目中添加依赖
uv add --editable /path/to/read-agent
```

然后：

```python
from literature_agent.models.document import Document
from literature_agent.agents.structure import StructureBuilder

# 直接使用数据模型或构建器
builder = StructureBuilder()
builder.process_heading(0, "Introduction", 1)
builder.process_heading(5, "Background", 2)
builder.process_non_heading(6)
print(builder.nodes)
```
