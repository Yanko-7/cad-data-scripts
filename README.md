# 项目结构说明

本项目是一套用于处理 CAD B-Rep（边界表示）数据的脚本集合，主要围绕 ABC 数据集，将 STEP/BRep 格式的 CAD 模型转换为 NPZ 格式的 Bézier 控制点特征，并完成去重、过滤、数据集划分等预处理流程。

---

## 根目录文件

| 文件 | 说明 |
|------|------|
| `utils.py` | **核心工具库**。基于 OpenCASCADE (OCC) 实现所有 B-Rep 几何处理功能，包括：读取/过滤 STEP 文件、将曲面/曲线转换为双三次 Bézier 表示、提取控制点特征（面、边、拓扑邻接）、形状预处理、有效性检验、水密性检验等。被绝大多数脚本直接 import。 |
| `dataset_loader.py` | **数据集加载工具**。提供 `load_dataset_fast()` 函数：读取 JSON 格式的数据集划分文件，并在指定目录中快速匹配到对应的 NPZ 文件绝对路径。被多个分析与过滤脚本 import。 |

---

## `acquisition/` — 数据获取

从外部来源下载或提取原始数据，生成后续处理所需的输入文件。

| 文件 | 说明 |
|------|------|
| `download_step_files.py` | 从 `step_files_urls.txt` 中读取 URL 列表，并发下载 STEP 文件，支持断点续传和多进度条显示。 |
| `step_files_urls.txt` | STEP 文件的下载 URL 列表，供 `download_step_files.py` 使用。 |
| `unzip.sh` | Shell 脚本，用于解压下载的压缩包。 |
| `build_abc1m_split_json.py` | 读取 `configs/abc1m_stems.json`（ABC-1M 数据集的 stem 列表），将文件名解析为统一的 ID 格式（`{8位ID}_{4位序号}`），输出 `configs/abc1m_split.json`。 |
| `extract_abc1m_brep.py` | 读取 ABC 数据集的 STEP 文件，逐个提取其中的有效实体（Solid），修复几何后以 `.brep` 格式保存到输出目录（按 file_id 分片存储）。 |
| `extract_abc1m_npz.py` | 与 `extract_abc1m_brep.py` 类似，但直接提取 Bézier 控制点特征，以压缩 NPZ 格式保存（跳过 BRep 中间格式）。 |
| `convert_deepcad_split.py` | 读取 DeepCAD 数据集的划分 JSON（`train_val_test_split.json`），将 `validation` 键重命名为 `val`，保存为标准格式（`deepcad.json`）。 |

---

## `processing/` — 数据处理与转换

将原始 STEP/BRep 文件批量转换为模型训练所需的 NPZ 特征文件。

| 文件 | 说明 |
|------|------|
| `extract_step_to_npz.py` | **主要特征提取脚本（STEP → NPZ）**。读取 STEP 文件，执行双三次 Bézier 转换和形状预处理，过滤不满足水密性/复杂度限制的模型，输出包含面控制点、边控制点及拓扑邻接信息的 NPZ 文件。使用 Pebble 进程池并行处理，支持超时保护。 |
| `extract_step_primitives.py` | **原始点特征提取（STEP → NPZ）**。与上一个脚本类似，但调用 `extract_primitive()` 提取原始几何采样点而非 Bézier 控制点。 |
| `extract_brep_to_npz.py` | **BRep → NPZ 特征提取**。输入为 `.brep` 格式文件（而非 STEP），调用 `extract_bicubic_features_dir()` 提取特征，适用于已经提取的实体文件。 |
| `split_step_solids.py` | 读取 STEP 文件，调用 `split_and_classify_step()` 将多实体 STEP 文件拆分为单实体并分类存储。 |
| `convert_step_to_ply.py` | 将 STEP 文件转换为 PLY 点云文件。先对 STEP 进行三角化网格化，再从表面采样 2000 个点，输出为 PLY 格式（用于可视化评估）。 |

---

## `filtering/` — 过滤与去重

对提取的 NPZ 数据集进行质量筛选和几何去重，确保训练数据的有效性与多样性。

| 文件 | 说明 |
|------|------|
| `validate_step_files.py` | 检验 STEP 文件的几何有效性（BRepCheck）与水密性（每条边是否都被两个面共享），输出统计结果。 |
| `filter_npz_by_topology.py` | **NPZ 拓扑过滤（主过滤脚本）**。加载 NPZ 文件，检验面数、边数、流形性（non-manifold 检测）、BBox 退化和重复几何等，输出通过过滤的文件 ID 列表（`configs/filtered_brep_abc1m_paths.json`）。 |
| `filter_step_by_validity.py` | 对 STEP 文件执行 Euler-Poincaré 检验、有效性检验和复杂度过滤（面数、每面边数、总边数上限），输出过滤后的文件 ID 列表（`abc_filtered_final.json`）。 |
| `dedup_cad_models.py` | **CAD 模型级去重**。对每个 NPZ 文件的点云做平移归一化 + 量化哈希（支持 8 种轴翻转），取最小哈希作为模型指纹，去除重复模型，输出去重后的数据集 JSON。 |
| `dedup_face_edge_geometry.py` | **面/边几何级去重**。对单个面或边的点集做平移 + PCA 旋转 + 归一化 + 翻转量化哈希，去除跨文件的重复面/边几何元素，输出唯一几何的数据字典（`.pkl`）。 |

---

## `dataset/` — 数据集组织与划分

将处理后的文件按面数分类存储，并生成 train/val/test 划分 JSON。

| 文件 | 说明 |
|------|------|
| `organize_and_split.py` | **两阶段数据集构建脚本**。阶段一：扫描源 NPZ 文件，根据文件名中的面数将其移动/复制到按面数区间命名的子目录（如 `01_Faces_0_to_10/`）；阶段二：扫描整理后的目录，随机划分为 train/val/test 并保存为 JSON。 |
| `dataset/pipeline_dedup_split.py` | **一体化 Pipeline**。对 NPZ 文件进行基于哈希的去重（支持 PCA 和翻转对称），按面数区间分类整理，最终生成数据集划分 JSON（`dataset_split.json`）。 |

---

## `analysis/` — 统计与分析

对数据集进行定量统计，辅助了解数据分布和模型训练配置。

| 文件 | 说明 |
|------|------|
| `stats_npz_distribution.py` | （代码已注释）扫描 NPZ 文件，统计面数/边数分布并生成可视化图表，保存为 `dataset_distribution.png`。 |
| `stats_step_distribution.py` | 分析目录中所有 STEP 文件的面数和边数分布，输出频率统计表格和折线图（PNG），并将详细结果保存为 JSON。 |
| `stats_token_ratio.py` | 将 NPZ 数据转换为序列表示后，统计 face token、edge token 和 text token 的数量占比，并建议各类型的复制倍率（用于训练时的数据均衡）。 |
| `plot_distributions.py` | 并行读取数据集中所有 NPZ 文件的面数/边数，绘制分 split（train/val/test）的分布直方图（含 KDE）、箱线图和面边联合散点图，保存为 PNG。 |
| `estimate_memory_usage.py` | 统计数据集中 `face_points` 数组的内存占用（单位 MB），输出平均值、中位数、最大值、最小值。 |

---

## `visualization/` — 可视化与测试

用于数据可视化、单文件调试和格式转换验证。

| 文件 | 说明 |
|------|------|
| `visualize_brep.py` | 从 NPZ 文件中读取 Bézier 面和曲线控制点，用 Bernstein 多项式求值，绘制 3D 点云散点图（面用颜色区分，边用黑线显示），批量保存为高分辨率 PNG。 |
| `convert_and_render_test.py` | 单文件调试脚本：加载一个 STEP 文件 → 转换为双三次 Bézier → 提取特征 → 保存 NPZ → 渲染为 HTML（调用 `inference.render_and_save_html`）。 |
| `test.step` | 用于调试的示例 STEP 文件。 |
| `dataset_distribution.png` | 数据集面数/边数分布的可视化图表（由 `stats_npz_distribution.py` 生成）。 |
| `render.html` | 由 `convert_and_render_test.py` 生成的 B-Rep 交互式 HTML 渲染结果。 |

---

## `configs/` — 数据集配置文件

存放各阶段生成或预置的数据集划分 JSON 文件，供各脚本通过相对路径 `configs/` 读取。

| 文件 | 说明 |
|------|------|
| `abc1m_stems.json` | ABC-1M 数据集的文件 stem 列表（按 train/val/test 划分），由外部工具生成，作为 `acquisition/` 脚本的输入。 |
| `abc1m_split.json` | 由 `build_abc1m_split_json.py` 生成，将 stem 解析为标准 ID 格式的划分 JSON。 |
| `brep_abc_data_split_6bit.json` | ABC 数据集 BRep 格式的 6bit 量化划分文件。 |
| `brepgen_deepcad_data_split_6bit.json` | DeepCAD 数据集的 6bit 量化划分文件，用于 BrepGen 训练。 |
| `filtered_brep_abc1m_paths.json` | 由 `filter_npz_by_topology.py` 生成，包含通过拓扑过滤的 ABC-1M 文件 ID。 |
| `filtered_brep_abc1M_data_split_6bit_paths.json` | ABC-1M 6bit 量化 + 过滤后的路径划分文件。 |
| `filtered_brep_deepcad_data_split_6bit_paths.json` | DeepCAD 6bit 量化 + 过滤后的路径划分文件。 |
| `matched_stems.json` | 匹配后的 stem 文件，用于对齐不同数据源的模型 ID。 |

---

## `archives/` — 压缩数据包

存放原始或预处理好的大体积数据压缩包，不参与脚本运行。

| 文件 | 说明 |
|------|------|
| `brepgen_abc.7z` | BrepGen 格式的 ABC 数据集压缩包。 |
| `brepgen_abc_test_ply.7z` | ABC 测试集的 PLY 点云压缩包。 |
| `solids_tol.7z` | 提取的实体 BRep 数据压缩包（含公差信息）。 |

---

## `ABC-dataset/` 和 `DEEPCAD-dataset/`

存放各数据集相关的独立工具脚本和原始划分文件，与主流程脚本相互独立。

| 文件 | 说明 |
|------|------|
| `ABC-dataset/get_dataset_info.py` | 获取 ABC 数据集元信息。 |
| `ABC-dataset/solve.py` | ABC 数据集相关处理辅助脚本。 |
| `ABC-dataset/abc_data_split_6bit.json` | ABC 数据集的 6bit 划分配置。 |
| `ABC-dataset/filtered_brep_abc_data_split_6bit_paths.json` | 过滤后的 ABC BRep 路径划分。 |
| `DEEPCAD-dataset/tmp.py` | 将 DeepCAD 的 PKL 格式划分文件转换为 JSON 格式。 |
| `DEEPCAD-dataset/train_val_test_split.json` | DeepCAD 原始 train/val/test 划分文件。 |
| `DEEPCAD-dataset/brepgen_deepcad_data_split_6bit.json` | BrepGen 使用的 DeepCAD 6bit 量化划分文件。 |

---

## 典型数据处理流程

```
ABC STEP 文件
    │
    ├─ acquisition/extract_abc1m_brep.py   → .brep 实体文件
    │       └─ processing/extract_brep_to_npz.py  → NPZ 特征
    │
    └─ processing/extract_step_to_npz.py   → NPZ 特征（直接从 STEP）
            │
            ├─ filtering/filter_npz_by_topology.py  → 过滤后的文件 ID JSON
            │
            ├─ filtering/dedup_cad_models.py         → 去重后的文件 ID JSON
            │
            ├─ dataset/organize_and_split.py         → 按面数分类 + train/val/test 划分
            │
            └─ analysis/plot_distributions.py        → 分布可视化
```
