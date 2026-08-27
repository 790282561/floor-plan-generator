---
name: floor-plan-generator
description: 根据户型图片按墙体、门、窗顺序分阶段识别并生成具有真实墙厚的墙体、门洞门扇及开启方向、窗洞窗框，并输出 CAD 图纸和必要标注。
---

# 建筑户型平面图生成 Skill

## 1. Skill目标
### 1.1 首要目标
根据户型图片、草图或 CAD 截图，识别建筑外轮廓、墙体、房间、门、窗和尺寸，建立统一毫米坐标模型，生成可编辑的建筑平面图。
所有墙体必须使用具有实际厚度的双线表达；门窗必须形成真实墙体洞口，不得只将符号叠加在连续墙线上。
### 1.2 适用范围
适用于户型图重绘、草图规范化、墙体 mask 边界提取、门洞门扇及开启弧生成、窗洞窗框生成、房间名称标注、尺寸标注和 CAD 输出。

### 1.3 脚本调用总原则
图形生成必须严格调用当前项目 `scripts` 目录中的实际程序，不得自行替换为不存在的脚本或绕过脚本直接猜画。图片任务必须依次调用 `processing.py`、`generate_by_pic.py`、`generate_doors.py` 和 `generate_windows.py`，CAD 生成链固定为“墙体 → 门 → 窗”。

### 1.4 测试与产物目录约定
所有测试、识别中间文件、预览图、DXF、JSON 报告和校验结果统一写入：
`D:\中建科技\009_自动化软件平台\outputs\<case_id>\`。
`<case_id>` 应使用输入图片名或明确的验证场景名，例如 `ref_pic_7`、`_straight_wall_validation`。项目目录中的 `scripts`、Skill 文档和源代码只保存程序，不再创建或使用 `floor-plan-generator\outputs` 作为运行产物目录。命令中的 `<preprocessed>`、`<walls.dxf>` 等路径必须解析到上述外部输出根目录下对应的案例目录。

## 2. 强制执行流程
### 2.1 输入解析与图片预处理
图片输入必须先运行：
`python scripts/processing.py <输入图片> --output-dir <预处理输出目录>`
然后读取该目录中的 `result.json`、`overlay.png`、`masks/walls.png`、`masks/doors.png`、`masks/windows.png`、`masks/dimensions.png`、`masks/room_labels.png` 和 `masks/other.png`。OCR 不可用或 `text=null` 时，必须根据房间边界、交通关系、洁具、橱柜灶台、阳台轮廓、家具及空间位置进行视觉语义推断，生成 `inferred_room_labels.json`；推断名称标记为 `INFERRED_VISUAL`，不得因此省略房间名称。

### 2.1b 图例先验加载
进入多模态识别前，必须先读取 `references/README.md`，并加载同目录下与类别对应的图例图片。当前图例文件为：`wall_legend.jpg`（墙体涂实）、`door_legend.jpg`（平开门及圆弧开启方向）、`window_legend.jpg`（普通窗及多条平行窗框线）、`sliding_door_legend.jpg`（推拉门及两扇搭接门板）、`wall_corner_legend.jpg`（墙角连接）、`window_corner_legend.jpg`（L 形连续转角窗框）、`annotated_floorplan_example.jpg` 和 `annotated_floorplan_example2.jpg`（完整户型图标注示例）。

图例只作为多模态分类和 OpenCV 分流的视觉先验，不得直接转换为 CAD 坐标，也不得覆盖当前户型图中的实际证据。识别时必须将图例中的符号与原图候选进行相似性比对，并结合墙体 ROI、像素位置、方向、拓扑关系和尺寸证据；图例与原图冲突时保留冲突记录并降低置信度。缺少某张图例时继续使用其余图例，不得因此中断整个流程。
### 2.1a 多模态识别与 OpenCV 分流
在进入墙、门、窗几何生成前，必须建立多模态候选层。多模态输入可以是当前对话中的户型图视觉识别结果，也可以是兼容本 Skill 数据契约的 JSON；不得把自然语言描述直接当作 CAD 坐标。多模态识别必须参考 `2.1b 图例先验加载` 中的实际图例，但最终候选仍以当前户型图为准。

`multimodal_fusion.py` 必须首先识别整张输入户型图的闭合建筑外轮廓，输出顶层 `building_outline`，至少包含 `bbox_px`、闭合 `polygon_px`、面积、周长、凹度、置信度、状态和证据来源，同时输出 `masks/building_outline.png`、`masks/building_footprint.png` 与 `building_outline_overlay.png`。该外轮廓由原图结构线及墙、门、窗预处理证据融合得到，只作为后续外边界缺口分析基准，不得修改或替代冻结的 `masks/walls.png`。

多模态识别至少输出 `multimodal.json`，每个候选包含：`id`、`class`（`building_outline`/`wall`/`door`/`window`）、`bbox_px` 或 `polygon_px`、`orientation`（可空）、`attributes`、`confidence`、`status`（`DETECTED`/`INFERRED`/`UNCERTAIN`）和 `reason`。墙体候选还应给出 `wall_role`；门候选应给出 `door_type`、`hinge_side`、`swing_direction`（不确定可空）；窗候选应给出 `window_type`。

随后由 OpenCV 按类别分别处理：`walls` 只生成墙体 mask、中心线/双边界和墙厚候选；`doors` 只在墙体 ROI 内寻找门洞、门扇和开启弧；`windows` 在墙体 ROI 内寻找窗洞/窗框，并排除已确认门候选及门弧范围。每类结果分别写入 `opencv_candidates`，保留像素坐标和证据来源，禁止不同类别共用一个未经分类的轮廓。

融合规则：同类且 IoU >= 0.35 的候选合并；多模态与 OpenCV 几何一致时提高置信度，不一致时保留冲突记录并降为 `UNCERTAIN`。凡图片中已提取出有效几何范围的墙、窗、门候选，都必须进入对应生成脚本并生成图元；`confidence >= 0.85` 标记为 `DETECTED`，低于该值或存在拓扑冲突时标记为 `INFERRED`/`ACCEPTED_IMAGE_ONLY` 并写入 `repair_queue`，但不得因此跳过图元。只有图片和预处理结果中均无该类候选时，才允许该类图元数量为 0。多模态层只提供识别证据，不绕过现有 CAD 生成脚本。
### 2.2 建筑外轮廓识别
识别整体边界、凹凸关系、转角、阳台、露台、门廊和附属空间，建立建筑外轮廓。轮廓必须来自整张输入图片，不得仅对墙体 mask 求外包矩形或凸包；必须保留真实凹口和高低错台。外轮廓完成后，才允许沿轮廓依次扣除已完成墙体范围和门 bounding box/门图元范围；剩余连续缺口只能作为窗候选证据，仍需由窗阶段结合原图窗线、方向及门窗互斥规则确认。
### 2.3 墙体识别
识别墙体两侧边界、中心位置、长度、方向、厚度、端点和连接关系。预处理时必须先对整体黑色前景执行总宽度缩减 10px 的腐蚀筛选：使用 11×11 核，使线条两侧各减少约 5px；腐蚀后完全消失的细线不得进入墙体 mask，仍有核心的粗黑墙体再膨胀回填并与原图相交，保留原始墙厚。该步骤用于剔除门扇、门槛、尺寸线、文字和家具细线。预处理完成后，`masks/walls.png` 是墙体几何的唯一参考标准，权重固定为 1.0；原图、多模态候选和历史墙体不得参与墙体几何。允许按用户要求进行做直、共线简化和使用 `data.json` 标准化墙宽，但不得按整个连通域包围框的横向高度或纵向宽度淘汰墙段；交叉连接的墙体必须通过行列投影拆分并全部保留。墙体阶段必须立即将生成墙体反向映射至像素空间与冻结 mask 比较，覆盖率不合格时不得标记为 `ready`。
### 2.4 墙体基线冻结
拾取 `masks/walls.png` 边线并完成做直、微小宽度归一化后生成的墙体 DXF，是本任务唯一且冻结的墙体基线。门、窗及房间名称阶段不得删除、切断、缩短、移动、炸开、闭合修补或重建任何 `WALLS`/`INNER_WALLS` 图元，只能读取墙体坐标用于门窗吸附并新增相应图层图元。每个后续阶段必须记录墙体基线路径、SHA-256，并校验墙体图元集合在阶段前后完全一致；窗阶段还必须记录其输入的含门 DXF 路径和 SHA-256。
### 2.5 墙体断口识别
识别门洞、窗洞及其他开口，确定所属墙段、方向、位置、宽度和开口边界。开口必须进入墙体几何模型。
### 2.6 门洞识别
识别单开门、双开门、推拉门、入户门、阳台门、卫生间门及其他门洞。平开门预处理 bounding box 必须是实际检测弧段的紧致外包框，由弧线起点、终点及弧段经过的必要极值点计算，禁止使用 `圆心 ± 半径` 的整圆外包框。对于归一化四分之一圆端点 `(1,0)`、`(0,1)`，bounding box 的对角点必须就是 `(1,0)`、`(0,1)`。门洞必须位于墙体上并形成真实开口。
### 2.7 门扇识别
识别门扇宽度、铰链位置、门扇方向和门型。门候选的 `boundingbox` 是尺寸硬边界：映射到 CAD 后，门洞宽度、门扇长度、开启弧半径及推拉门板总范围均不得超过 boundingbox 在所属墙向上的映射尺寸。墙线吸附、门槛匹配和拓扑修正只能缩小或保持尺寸，禁止扩大。平开门使用门扇线和开启弧表达，推拉门使用两扇相互搭接的门板线框表达。
### 2.8 门开启方向识别
结合 wall_orientation、swing_direction 和 overlay.png 判断开启方向。无法可靠判断时标记 INFERRED，不得伪造为确定方向。
### 2.9 窗识别
识别普通窗、落地窗、飘窗、转角窗、推拉窗及其他窗洞。普通窗必须检测到同一墙段上四条基本等长、相互平行且间距合理的直线，不含门扇开启圆弧；少于或多于无法归并为明确四线组时，不得定为窗。转角窗必须检测到四条连续的 L 形线：每条线均由水平段和竖直段在同一墙角连续相接；必须作为一个 `corner_window` 组记录。未检测到完整四线证据时，不得生成普通窗或转角窗图元。
### 2.9a 门窗互斥规则
四分之一圆弧与从铰链出发的门扇直线是平开门强证据。任一窗候选与门弧候选区域重叠时，门的分类优先。由于门阶段先执行，窗阶段必须读取并保留既有 `DOORS` 图元，抑制与门候选 bounding box、门扇或门弧重叠的窗候选；不得删除、移动、缩放或覆盖已生成的门图元。门弧所在位置不得同时存在 `WINDOWS` 图元。
### 2.10 尺寸与比例校准
尺寸优先级为 USER > 图片明确标注 > 多个已知尺寸计算值 > 比例推定值。已知尺寸时使用 scale = real_length / pixel_length，并交叉校验。坐标单位统一为 mm，原点为 (0,0)。
### 2.11 置信度判断
低置信度、候选重叠或无法由拓扑确认的对象必须标记 `INFERRED`、`UncertainObject` 或 `ACCEPTED_IMAGE_ONLY`，同时仍按图片几何生成图元；不得把它描述为已经拓扑确认的确定对象。
### 2.12 建筑规则校核
校核外轮廓、墙体连续性、墙厚、房间闭合性、门窗位置、洞口关系、开启方向、比例和交通关系。
### 2.13 生成结构化数据
生成 Wall、Opening、Door、Window、Room 和 UncertainObject 数据。用户要求优先于图片明确尺寸，图片明确尺寸优先于几何推定。Room 必须包含房间名称、文字位置、文字高度、文字样式和来源置信度。
### 2.14 脚本驱动的 CAD 生成
不得直接由 Skill 手工绘制图形，必须按图片阶段调用对应脚本：预处理调用 `processing.py`，墙体调用 `generate_by_pic.py`，门调用 `generate_doors.py`，窗户调用 `generate_windows.py`。墙体 DXF 必须作为门阶段输入，含门 DXF 必须作为窗阶段输入，窗阶段输出才是最终 DXF。
### 2.15 绘后检查
每个生成脚本都必须读取其输出的同名 JSON 报告并完成阶段验收：墙体阶段记录墙线、多段线和审计信息；门阶段记录 `accepted_doors`、房间名称和墙体完整几何签名前后对比；最终窗阶段记录门窗图元保留情况、墙体完整几何签名，并将最终 DXF 墙体重新映射为像素 mask，与冻结的 `masks/walls.png` 比较。允许做直和墙宽标准化带来的半个标准墙厚容差，但 `buffered_mask_recall` 不得低于 0.90，`buffered_cad_precision` 不得低于 0.70；否则视为严重轮廓偏差或墙体丢失，必须设置 `final_output_qualified: false`、`stage_status: needs_repair` 并输出墙体 mask 对比图。无论候选为空或校验失败，仍保留 DXF/PNG/JSON 诊断产物和 `repair_queue`，但不得把未通过结果描述为合格最终图。
### 2.16 局部纠错
只修改对应门窗 bounding box 及必要墙厚缓冲范围内的局部对象，并重新检查相邻墙体、门窗、房间和尺寸关系。不得全量炸开、转换、删除或重建全部墙体多段线，不得因局部纠错改变无关区域。被误识别为墙的门槛、门扇线只能在对应门候选范围内剔除。

### 2.17 预处理证据的使用边界
`processing.py` 生成的 `result.json`、`overlay.png` 和 masks 是识别证据与降级输入。门的位置、门型和开启方向优先由 `generate_doors.py` 根据门阶段图片重新计算；差分检测无候选时，必须回退使用 `result.json` 的门候选生成 `ACCEPTED_IMAGE_ONLY` 图元。窗户位置和几何随后由 `generate_windows.py` 根据窗阶段图片重新计算；主检测器无候选时，必须回退使用 `result.json` 的窗候选生成 `ACCEPTED_IMAGE_ONLY` 图元，同时保留上一步已生成的门。房间名称优先使用用户名称，其次使用明确 OCR；OCR 不可用或为空时，必须根据图像布局和室内构件进行视觉推断，写入 `inferred_room_labels.json` 并在门阶段落图，窗阶段必须原样保留。候选框可作为降级几何范围，但必须在报告中标记来源和不确定性。

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
门候选必须匹配所属墙体和墙厚，但冻结墙体不得在门阶段再次切割；门扇、门弧及门洞语义记录新增到 `DOORS` 图层和报告中。
### 5.4 门扇规则
平开门表达门扇、铰链和开启弧；双开门表达两扇门；推拉门表达两扇相互搭接的门板。
### 5.4a 门 boundingbox 尺寸约束
每个门必须记录紧致弧段像素 bounding box、`arc_start_deg`、`arc_end_deg`、映射后的 `bbox_max_width_mm`、生成前宽度、最终生成宽度和是否发生截断。最终门洞起终点必须位于紧致 bounding box 映射区间内，`generated_width_mm <= bbox_max_width_mm`；平开门的门扇长度和圆弧半径必须等于最终门洞宽度，推拉门板不得越出最终门洞。不得使用整圆范围、附近墙端点、门槛封边或常用门宽扩大检测尺寸。
### 5.5 门开启方向规则
开启方向由铰链、门扇和开启弧共同表达，并应与图片和房间交通关系一致。
### 5.6 窗洞规则
窗候选必须匹配所属墙体，窗框宽度继承墙厚并吸附墙体正交边界；冻结墙体不得在窗阶段再次切割或重建。
### 5.6a 转角窗规则
转角窗必须由四条连续 L 形线组成，每条 L 形线都包含互相垂直且在墙角相接的水平段和竖直段。CAD 中两侧窗段均位于 `WINDOWS` 图层，报告中以同一个 `corner_window` 组关联并记录 `evidence_line_count: 4`。只有一侧窗段、两侧不在同一转角、线条少于四条或包含门弧时，不得标记为转角窗。
### 5.7 房间闭合规则
房间应由连续墙体围合形成闭合空间，不得存在墙体穿过房间、异常狭窄空间或无依据分隔。

## 6. CAD绘制规则
### 6.1 图层
至少使用 WALLS、INNER_WALLS、DOORS、WINDOWS、DIMENSIONS 和 ROOM_NAMES 图层；辅助检测图形不得替代正式图形。
### 6.2 墙体
使用闭合双线墙体轮廓或等效有厚度几何对象，正确处理墙体连接和门窗开洞；不得以独立中心线作为最终墙体。
### 6.3 门
在 `DOORS` 图层绘制门扇、门弧和推拉门板；不得修改冻结的墙体图元。
### 6.4 窗
在 `WINDOWS` 图层绘制窗框；窗框位于对应墙体范围内且方向、宽度与墙体一致，不得修改冻结的墙体图元。
### 6.5 文字
房间名称必须生成到 `ROOM_NAMES` 图层，统一使用 `simhei.ttf`（`C:\Windows\Fonts\simhei.ttf`）字体、统一文字样式和高度，文字基点位于对应房间中心或经校核的室内空白区域，并使用居中对齐。优先级为用户指定名称 > OCR 明确文字 > 视觉语义推断。OCR 不可用或不确定时不得停止落图，应根据空间边界、家具、洁具、灶台、阳台及交通关系推断名称，标记 `INFERRED_VISUAL` 和置信度。文字不得覆盖墙体、门窗、尺寸和其他房间名称，不得出现乱码。
### 6.6 尺寸
按可靠尺寸建立比例并绘制必要尺寸。推定尺寸必须与确定尺寸区分，尺寸线应离开建筑墙体并保持清晰间隙。

## 7. 纠错机制
采用“识别—建模—校核—局部修正—再次校核”闭环。无法可靠判断时保留候选并标记 INFERRED，必要时说明不确定区域，不得凭空制造复杂结构。

## 7.1 实际脚本调用流程

### 图片输入流程

1. 预处理：`python scripts/processing.py <source.png> --output-dir D:\中建科技\009_自动化软件平台\outputs\<case_id>\preprocessed`
2. 多模态/OpenCV 融合：`python scripts/multimodal_fusion.py <source.png> --preprocessed-dir D:\中建科技\009_自动化软件平台\outputs\<case_id>\preprocessed --output D:\中建科技\009_自动化软件平台\outputs\<case_id>\preprocessed\multimodal.json`；先读取并检查 `building_outline` 及其 mask/overlay，再按 `wall`、`door`、`window` 分流并处理冲突。
3. 墙体生成：`python scripts/generate_by_pic.py <source.png> D:\中建科技\009_自动化软件平台\outputs\<case_id>\walls.dxf --data scripts/data.json --wall-mask D:\中建科技\009_自动化软件平台\outputs\<case_id>\preprocessed\masks\walls.png --overall-width-mm <总宽> --overall-height-mm <总高> --wall-bbox <x1> <y1> <x2> <y2> --horizontal-chain-mm <横向尺寸链> --vertical-chain-top-down-mm <纵向尺寸链>`；mask 固定按 1.0 权重作为唯一墙体几何依据。
4. 门及房间名称生成：`python scripts/generate_doors.py <含门图片> <墙体阶段图片> <walls.dxf> <walls_doors.dxf> --data scripts/data.json --overall-width-mm <总宽> --overall-height-mm <总高> --image-wall-bbox <x1> <y1> <x2> <y2> --preprocessing-result <preprocessed/result.json> --inferred-room-labels <inferred_room_labels.json>`
5. 窗户及最终图生成：`python scripts/generate_windows.py <含窗图片> <walls_doors.dxf> <final.dxf> --data scripts/data.json --overall-width-mm <总宽> --overall-height-mm <总高> --image-wall-bbox <x1> <y1> <x2> <y2> --preprocessing-result <preprocessed/result.json>`

墙体脚本输出的 DXF 和 JSON 报告必须先读取并记录，再原样传给 `generate_doors.py`；门阶段输出的 `walls_doors.dxf` 和同名 JSON 必须读取后传给 `generate_windows.py`。后续报告必须记录原始墙体基线的绝对路径和 SHA-256，窗阶段还必须记录含门 DXF 的绝对路径和 SHA-256。若墙体尚未闭合，门和窗脚本仍继续生成图元，但不得替换为历史闭合墙模型，也不得全量将墙体 `LWPOLYLINE` 转换为 `LINE`。原图门线和窗线不是中断条件。尺寸链、总宽、总高和墙体外框不能凭空填写，必须来自用户、图纸标注或经过复核的图像测量。

`generate_by_pic.py` 只允许把 `masks/walls.png` 中逐像素拾取的原始边界写入墙体 DXF。独立 `LINE` 是原始 mask 边界的合法表达；不得为了闭合验收把边界转换或重建为 `LWPOLYLINE`，不得以墙体闭合或拓扑状态阻断后续门、窗阶段。

### 7.2 非致命阶段与修复队列
各阶段采用“继续识别、延后阻断”策略：

1. 发现墙体断裂、悬空、重复或未闭合时，记录对象 ID、像素坐标、问题类型、可能修复方式和置信度，加入 `repair_queue`。
2. 继续运行不依赖闭合墙体的多模态候选提取、OpenCV 门窗候选提取、OCR 和尺寸证据整理；不得伪造墙体拓扑已通过。
3. 后续脚本必须读取 `stage_status`，但不得用它决定是否执行。`ready` 时优先切真实洞口；`needs_repair` 时仍必须生成门窗图元，无法可靠切洞时按图片坐标生成 `ACCEPTED_IMAGE_ONLY` 图元，并把依赖关系写入报告。
4. 修复队列清空且重新校核通过后，才允许把阶段状态改为 `ready`，再进行正式门窗几何和最终 CAD 验收。
5. 任一阶段失败都要返回结构化状态和已有产物路径；只有输入不可读、输出无法写入或数据损坏等致命错误才允许结束进程。

### 脚本职责边界

- `processing.py` 只负责图片预处理和识别证据输出。
- `generate_by_pic.py` 只负责图片墙体及墙体尺寸 DXF。
- `generate_doors.py` 只读取冻结墙体用于吸附，生成门图形并写入房间名称；必须保持全部墙体图元不变。
- `generate_windows.py` 只读取冻结墙体用于吸附并生成窗框；必须保持全部墙体、既有 `DOORS`、`ROOM_NAMES` 和无关图元不变。

`generate_doors.py` 提供 `--inferred-room-labels` 入口，将视觉读取结果写入 `ROOM_NAMES` 图层并在阶段 JSON 中记录名称、像素位置、CAD 位置、文字高度、来源和置信度。OCR 不可用不得跳过该参数；应先生成视觉推断 JSON，再完成门及房间名称阶段。随后执行的 `generate_windows.py` 不得删除、改名或移动这些文字。

不得使用不存在的 `generate.py`，不得让 `generate_by_pic.py` 代替门窗脚本，也不得让门窗脚本跳过前置墙体结果。

## 8. 输入方式与使用示例
图片输入：请根据这张户型图生成墙体、门和窗，墙体使用双线墙，并添加必要尺寸标注。必须按 processing.py → generate_by_pic.py → generate_doors.py → generate_windows.py 顺序执行。

## 9. 默认参数
单位为 mm。`scripts/generate_by_pic.py` 中的总宽、总高、墙体外包框和尺寸链是特定参考图的默认值，不得直接套用于新图片。每个新图片任务必须重新确认并传入 `--overall-width-mm`、`--overall-height-mm`、`--wall-bbox`、`--horizontal-chain-mm` 和 `--vertical-chain-top-down-mm`；窗户和门阶段对应传入 `--image-wall-bbox`。`scripts/data.json` 中的 `wall_width` 只作为允许墙厚候选，不能替代图片校准。

## 10. 禁止事项
- 使用单线代替墙体
- 忽略或随意改变墙体厚度、房间数量和建筑外轮廓
- 未执行图片预处理就直接生成
- 颠倒“墙体 → 门 → 窗”链路，或让窗阶段先于门阶段执行
- 墙体生成时不传入 `masks/walls.png`，或使用原图及任何其他证据混合、覆盖 mask 墙体边界
- 在拾取 mask 墙边线后继续做过滤、简化、吸附、合并、矩形化、墙厚归一化、闭合修补或拓扑重建
- 预处理墙体时跳过总宽度缩减 10px 的粗线筛选，或直接把腐蚀后的缩小墙厚作为最终墙体
- 因低置信度、`needs_repair` 或墙体匹配失败而跳过已有图片候选的门窗图元
- 将门窗放置在墙体之外
- 在门窗阶段删除、切断、缩短、移动、炸开或重建冻结墙体图元
- 门洞与墙体不对应，或窗与墙体重叠
- 在同一门弧区域同时保留门和窗图元
- 窗阶段删除、移动、缩放、覆盖既有 `DOORS` 或 `ROOM_NAMES` 图元
- 房间名称落图时修改任意墙体图元，或最终出图前未比较冻结 wall mask 与 CAD 墙体覆盖率
- 将门洞、门扇、门弧或推拉门板扩大到门候选 boundingbox 之外
- 使用 `圆心 ± 半径` 的整圆方框代替实际门弧的紧致 bounding box
- 把单段普通窗或门弧误判为转角窗
- 在没有完整四条平行线或四条连续 L 形线证据时生成窗、转角窗图元
- 将推定尺寸描述为准确尺寸
- 为追求视觉效果破坏建筑几何关系
- 在门窗阶段复用历史墙体 DXF、其他图片墙体模型或预置洞口墙体
- 全量炸开、转换或重建当前图片墙体，导致墙体基线整体变化
- 使用脚本内置的参考图尺寸、墙体外框或尺寸链处理新图片
- 未读取阶段 JSON 报告或未完成 PNG/CAD 检查就进入下一阶段
