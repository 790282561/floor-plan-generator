---
name: floor-plan-generator
description: 根据户型图片分阶段识别并生成具有真实墙厚的墙体、窗洞窗框、门洞门扇及开启方向，并输出 CAD 图纸和必要标注。
---

# 建筑户型平面图生成 Skill

## 1. Skill目标
### 1.1 首要目标
根据户型图片、草图或 CAD 截图，识别建筑外轮廓、墙体、房间、门、窗和尺寸，建立统一毫米坐标模型，生成可编辑的建筑平面图。
所有墙体必须使用具有实际厚度的双线表达；门窗必须形成真实墙体洞口，不得只将符号叠加在连续墙线上。
### 1.2 适用范围
适用于户型图重绘、草图规范化、墙体拓扑重建、窗洞窗框生成、门洞门扇及开启弧生成、房间名称标注、尺寸标注和 CAD 输出。

### 1.3 脚本调用总原则
图形生成必须严格调用当前项目 `scripts` 目录中的实际程序，不得自行替换为不存在的脚本或绕过脚本直接猜画。图片任务必须依次调用 `processing.py`、`generate_by_pic.py`、`generate_windows.py` 和 `generate_doors.py`。

## 2. 强制执行流程
### 2.1 输入解析与图片预处理
图片输入必须先运行：
`python scripts/processing.py <输入图片> --output-dir <预处理输出目录>`
然后读取该目录中的 `result.json`、`overlay.png`、`masks/walls.png`、`masks/doors.png`、`masks/windows.png`、`masks/dimensions.png`、`masks/room_labels.png` 和 `masks/other.png`。OCR 不可用时，text=null 的候选不得视为确定文字。
### 2.1a 多模态识别与 OpenCV 分流
在进入墙、窗、门几何生成前，必须建立多模态候选层。多模态输入可以是当前对话中的户型图视觉识别结果，也可以是兼容本 Skill 数据契约的 JSON；不得把自然语言描述直接当作 CAD 坐标。

多模态识别至少输出 `multimodal.json`，每个候选包含：`id`、`class`（`wall`/`window`/`door`）、`bbox_px` 或 `polygon_px`、`orientation`（可空）、`attributes`、`confidence`、`status`（`DETECTED`/`INFERRED`/`UNCERTAIN`）和 `reason`。墙体候选还应给出 `wall_role`；门候选应给出 `door_type`、`hinge_side`、`swing_direction`（不确定可空）；窗候选应给出 `window_type`。

随后由 OpenCV 按类别分别处理：`walls` 只生成墙体 mask、中心线/双边界和墙厚候选；`windows` 只在墙体 ROI 内寻找窗洞/窗框；`doors` 只在墙体 ROI 内寻找门洞、门扇和开启弧。每类结果分别写入 `opencv_candidates`，保留像素坐标和证据来源，禁止不同类别共用一个未经分类的轮廓。

融合规则：同类且 IoU >= 0.35 的候选合并；多模态与 OpenCV 几何一致时提高置信度，不一致时保留冲突记录并降为 `UNCERTAIN`。`confidence >= 0.85` 且无拓扑冲突才可自动进入对应生成脚本；`0.65–0.85` 必须局部复核；低于 `0.65` 或类别冲突必须进入 `UncertainObject`。多模态层只提供识别证据，不绕过现有 CAD 生成脚本。
### 2.2 建筑外轮廓识别
识别整体边界、凹凸关系、转角、阳台、露台、门廊和附属空间，建立建筑外轮廓。
### 2.3 墙体识别
识别墙体两侧边界、中心位置、长度、方向、厚度、端点和连接关系，并用 overlay.png 与 walls.png 复核。不得将墙体简化为单根中心线。
### 2.4 墙体拓扑重建
重建水平墙、垂直墙、L 型、T 型、十字型和转角连接，保证连续、无断裂、重复或悬空。
### 2.5 墙体断口识别
识别门洞、窗洞及其他开口，确定所属墙段、方向、位置、宽度和开口边界。开口必须进入墙体几何模型。
### 2.6 门洞识别
识别单开门、双开门、推拉门、入户门、阳台门、卫生间门及其他门洞。门洞必须位于墙体上并形成真实开口。
### 2.7 门扇识别
识别门扇宽度、铰链位置、门扇方向和门型。平开门使用门扇线和开启弧表达，推拉门使用两扇相互搭接的门板线框表达。
### 2.8 门开启方向识别
结合 wall_orientation、swing_direction 和 overlay.png 判断开启方向。无法可靠判断时标记 INFERRED，不得伪造为确定方向。
### 2.9 窗识别
识别普通窗、落地窗、飘窗、转角窗、推拉窗及其他窗洞。窗必须位于墙体内部，并形成真实窗洞和窗框。
### 2.10 尺寸与比例校准
尺寸优先级为 USER > 图片明确标注 > 多个已知尺寸计算值 > 比例推定值。已知尺寸时使用 scale = real_length / pixel_length，并交叉校验。坐标单位统一为 mm，原点为 (0,0)。
### 2.11 置信度判断
低置信度、候选重叠或无法由拓扑确认的对象必须标记 INFERRED 或 UncertainObject，不得直接当作确定门窗或墙体。
### 2.12 建筑规则校核
校核外轮廓、墙体连续性、墙厚、房间闭合性、门窗位置、洞口关系、开启方向、比例和交通关系。
### 2.13 生成结构化数据
生成 Wall、Opening、Door、Window、Room 和 UncertainObject 数据。用户要求优先于图片明确尺寸，图片明确尺寸优先于几何推定。Room 必须包含房间名称、文字位置、文字高度、文字样式和来源置信度。
### 2.14 脚本驱动的 CAD 生成
不得直接由 Skill 手工绘制图形，必须按图片阶段调用对应脚本：预处理调用 `processing.py`，墙体调用 `generate_by_pic.py`，窗户调用 `generate_windows.py`，门调用 `generate_doors.py`。各阶段的 DXF 输出必须作为下一阶段的输入。
### 2.15 绘后检查
每个生成脚本都必须读取其输出的同名 JSON 报告并完成阶段验收：墙体阶段记录 `wall_geometry_valid`、`wall_boundary_lines`、`wall_open_polylines`、`wall_closed_polylines` 和 `audit_errors`；窗户阶段记录 `accepted_windows`、`topology.wall_overlaps_after` 和 `audit_errors`；门阶段记录 `accepted_doors`、`topology.wall_overlaps_after`、`topology.door_geometry_collision_count` 和 `audit_errors`。无论原图窗线、门线、候选为空、候选被拒绝、墙体不封闭、存在开放折线或审计问题，均不得中途停止计算：必须保留当前 DXF/PNG/JSON，写入 `repair_queue`，标记 `stage_status: needs_repair`，并继续后续阶段和报告生成。只有输入不可读、输出无法写入或数据损坏等致命错误才允许结束进程。最终交付前再将未修复问题作为阻断项，不得宣称 CAD 几何合格。
### 2.16 局部纠错
只修改对应局部对象，并重新检查相邻墙体、门窗、房间和尺寸关系。不得因局部纠错改变无关区域。

### 2.17 预处理证据的使用边界
`processing.py` 生成的 `result.json`、`overlay.png` 和 masks 是识别证据与审计记录，不是门窗几何的直接输入。窗户位置和几何必须由 `generate_windows.py` 根据图片和墙体 DXF 重新计算；门的位置、门型和开启方向必须由 `generate_doors.py` 根据当前含门图片、上一阶段图片和墙窗 DXF 重新计算。房间名称必须使用 OCR 明确识别的文字或用户提供的名称，并转换为统一的 CAD 文字实体；OCR 为 null 或低置信度时不得擅自补写。不得把候选框直接当作确定文字或 CAD 门窗。

## 3. 数据模型
### 3.1 Wall
包含 id、start、end、thickness、type、openings、source 和 confidence；最终必须生成实际厚度的双线墙体。
### 3.2 Opening
包含 id、wall_id、position、width、type、source 和 confidence，表示墙体上的真实开口。
### 3.3 Door
包含 opening_id、type、hinge_position、width、wall_orientation、swing_direction 和 opening_arc。
### 3.4 Window
包含 opening_id、type、position、width、frame_lines、source 和 confidence。
### 3.5 Room
包含 id、name、x、y、width、height、boundary_walls、label_position、text_height、text_style、source 和 confidence；房间应由墙体围合形成闭合空间。`name` 必须与图片明确文字或用户要求一致。
### 3.6 UncertainObject
记录模糊、重叠或无法确认的墙体、门窗、文字和尺寸候选，包含 geometry、reason、confidence 和 INFERRED 状态。

### 3.7 MultimodalCandidate
包含 `id`、`class`、`bbox_px`/`polygon_px`、`attributes`、`source`、`confidence`、`status`、`opencv_evidence_ids` 和 `conflict_ids`。该对象用于把图像语义识别稳定地交给 OpenCV 分类别提取，不能直接作为 DXF 实体。

## 4. 信息可信度
### 4.1 USER
用户明确提供的墙厚、图片校准尺寸、门窗尺寸或布局关系，优先级最高。
### 4.2 DETECTED
从图片中明确识别出的墙体、门窗、房间文字、尺寸和几何关系。
### 4.3 INFERRED
根据比例、连续性、对称关系、房间功能或建筑常识推定的信息，必须显式标记。
### 4.4 Confidence
低置信度或与其他类别重叠的候选必须进入不确定对象流程。优先级为 USER > DETECTED > INFERRED。

## 5. 建筑几何规则
### 5.1 双线墙规则
所有墙体必须由两条平行边界线构成并具有实际墙厚，禁止单线墙。
### 5.2 墙体连接规则
正确处理 L、T、十字连接、转角和外墙内墙连接，不得断裂、错位、重复或悬空。
### 5.3 门洞规则
先在所属墙体中创建真实门洞，再放置门扇和开启弧；禁止在完整墙体上直接叠加门符号。
### 5.4 门扇规则
平开门表达门扇、铰链和开启弧；双开门表达两扇门；推拉门表达两扇相互搭接的门板。
### 5.5 门开启方向规则
开启方向由铰链、门扇和开启弧共同表达，并应与图片和房间交通关系一致。
### 5.6 窗洞规则
先中断墙体形成真实窗洞，再生成窗框；窗必须位于所属墙体内部，不得与墙体重叠。
### 5.7 房间闭合规则
房间应由连续墙体围合形成闭合空间，不得存在墙体穿过房间、异常狭窄空间或无依据分隔。

## 6. CAD绘制规则
### 6.1 图层
至少使用 WALLS、INNER_WALLS、DOORS、WINDOWS、DIMENSIONS 和 ROOM_NAMES 图层；辅助检测图形不得替代正式图形。
### 6.2 墙体
使用闭合双线墙体轮廓或等效有厚度几何对象，正确处理墙体连接和门窗开洞；不得以独立中心线作为最终墙体。
### 6.3 门
在 DOORS 图层绘制门扇、门弧和推拉门板；门洞必须切断所属墙体并与两端封边连接。
### 6.4 窗
在 WINDOWS 图层绘制窗框；窗洞必须切断所属墙体，窗框位于洞口内部且方向与墙体一致。
### 6.5 文字
房间名称必须生成到 `ROOM_NAMES` 图层。所有房间名称使用统一的文字样式和统一文字高度，文字基点位于对应房间几何中心或经校核的室内空白区域，并使用居中对齐。文字内容必须与图片 OCR 明确结果或用户指定名称一致，不得擅自翻译、缩写或改名；OCR 不确定时标记 `INFERRED` 并停止自动落图。文字不得覆盖墙体、门窗、尺寸和其他房间名称，不得出现乱码。
### 6.6 尺寸
按可靠尺寸建立比例并绘制必要尺寸。推定尺寸必须与确定尺寸区分，尺寸线应离开建筑墙体并保持清晰间隙。

## 7. 纠错机制
采用“识别—建模—校核—局部修正—再次校核”闭环。无法可靠判断时保留候选并标记 INFERRED，必要时说明不确定区域，不得凭空制造复杂结构。

## 7.1 实际脚本调用流程

### 图片输入流程

1. 预处理：`python scripts/processing.py <source.png> --output-dir <preprocessed>`
2. 多模态/OpenCV 融合：`python scripts/multimodal_fusion.py <source.png> --preprocessed-dir <preprocessed> --output <preprocessed/multimodal.json>`；读取 `multimodal.json`，按 `wall`、`window`、`door` 分流并处理冲突。
3. 墙体生成：`python scripts/generate_by_pic.py <source.png> <walls.dxf> --data scripts/data.json --overall-width-mm <总宽> --overall-height-mm <总高> --wall-bbox <x1> <y1> <x2> <y2> --horizontal-chain-mm <横向尺寸链> --vertical-chain-top-down-mm <纵向尺寸链>`
4. 窗户生成：`python scripts/generate_windows.py <含窗图片> <walls.dxf> <walls_windows.dxf> --data scripts/data.json --overall-width-mm <总宽> --overall-height-mm <总高> --image-wall-bbox <x1> <y1> <x2> <y2> --preprocessing-result <preprocessed/result.json>`
5. 门生成：`python scripts/generate_doors.py <含门图片> <上一阶段图片> <walls_windows.dxf> <final.dxf> --data scripts/data.json --overall-width-mm <总宽> --overall-height-mm <总高> --image-wall-bbox <x1> <y1> <x2> <y2> --preprocessing-result <preprocessed/result.json>`

墙体脚本输出的 DXF 和 JSON 报告必须先读取并记录，再传给 `generate_windows.py`；若墙体尚未闭合，窗户和门脚本以“候选识别/待修复模式”运行，输出候选报告并继续门阶段，不得把结果标记为最终 CAD 合格。原图窗线和门线不是中断条件。尺寸链、总宽、总高和墙体外框不能凭空填写，必须来自用户、图纸标注或经过复核的图像测量。

`generate_by_pic.py` 无论识别到实心墙带还是细线墙体，都应尝试重建为闭合 `LWPOLYLINE`。禁止将独立 `LINE` 或未闭合 `LWPOLYLINE` 当作最终墙体；无法闭合时脚本应写出带 `stage_status: needs_repair`、问题位置和 `repair_queue` 的中间 DXF/JSON/PNG，并返回可被上层流程记录的非致命阶段状态，不得因单个墙段问题直接结束整条识别流程。

### 7.2 非致命阶段与修复队列
各阶段采用“继续识别、延后阻断”策略：

1. 发现墙体断裂、悬空、重复或未闭合时，记录对象 ID、像素坐标、问题类型、可能修复方式和置信度，加入 `repair_queue`。
2. 继续运行不依赖闭合墙体的多模态候选提取、OpenCV 门窗候选提取、OCR 和尺寸证据整理；不得伪造墙体拓扑已通过。
3. 后续脚本必须读取 `stage_status`。`ready` 时可执行正式洞口切割；`needs_repair` 时只能输出候选/预览/诊断结果，并把依赖关系写入报告。
4. 修复队列清空且重新校核通过后，才允许把阶段状态改为 `ready`，再进行正式门窗几何和最终 CAD 验收。
5. 任一阶段失败都要返回结构化状态和已有产物路径；只有输入不可读、输出无法写入或数据损坏等致命错误才允许结束进程。

### 脚本职责边界

- `processing.py` 只负责图片预处理和识别证据输出。
- `generate_by_pic.py` 只负责图片墙体及墙体尺寸 DXF。
- `generate_windows.py` 只负责在已校核墙体 DXF 上切窗洞并生成窗框。
- `generate_doors.py` 只负责在已校核墙体和窗户 DXF 上切门洞并生成门图形。

当前脚本链尚未在 `generate_by_pic.py`、`generate_windows.py` 或 `generate_doors.py` 中提供房间名称落图入口；在该能力补入现有图片生成脚本前，不得宣称“房间名称已生成完成”。房间文字实现必须继续使用现有图片流程，并输出 `ROOM_NAMES` 图层及对应 JSON 记录，不得恢复文字尺寸独立生成路线。

不得使用不存在的 `generate.py`，不得让 `generate_by_pic.py` 代替门窗脚本，也不得让门窗脚本跳过前置墙体结果。

## 8. 输入方式与使用示例
图片输入：请根据这张户型图生成墙体、窗户和门，墙体使用双线墙，并添加必要尺寸标注。必须按 processing.py → generate_by_pic.py → generate_windows.py → generate_doors.py 顺序执行。

## 9. 默认参数
单位为 mm。`scripts/generate_by_pic.py` 中的总宽、总高、墙体外包框和尺寸链是特定参考图的默认值，不得直接套用于新图片。每个新图片任务必须重新确认并传入 `--overall-width-mm`、`--overall-height-mm`、`--wall-bbox`、`--horizontal-chain-mm` 和 `--vertical-chain-top-down-mm`；窗户和门阶段对应传入 `--image-wall-bbox`。`scripts/data.json` 中的 `wall_width` 只作为允许墙厚候选，不能替代图片校准。

## 10. 禁止事项
- 使用单线代替墙体
- 忽略或随意改变墙体厚度、房间数量和建筑外轮廓
- 未执行图片预处理就直接生成
- 把低置信度候选当成确定的门、窗、文字或尺寸
- 将门窗放置在墙体之外
- 只叠加门窗符号而不创建真实墙体洞口
- 门洞与墙体不对应，或窗与墙体重叠
- 将推定尺寸描述为准确尺寸
- 为追求视觉效果破坏建筑几何关系
- 使用脚本内置的参考图尺寸、墙体外框或尺寸链处理新图片
- 未读取阶段 JSON 报告或未完成 PNG/CAD 检查就进入下一阶段
