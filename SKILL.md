---
name: floor-plan-generator
description: 根据户型图片或房间尺寸识别并生成具有真实墙厚的墙体、窗洞窗框、门洞门扇及开启方向，并输出 CAD 图纸和必要标注。
---

# 建筑户型平面图生成 Skill

## 1. Skill目标
### 1.1 首要目标
根据户型图片、草图、CAD 截图或文字尺寸，识别建筑外轮廓、墙体、房间、门、窗和尺寸，建立统一毫米坐标模型，生成可编辑的建筑平面图。
所有墙体必须使用具有实际厚度的双线表达；门窗必须形成真实墙体洞口，不得只将符号叠加在连续墙线上。
### 1.2 适用范围
适用于户型图重绘、草图规范化、墙体拓扑重建、窗洞窗框生成、门洞门扇及开启弧生成、房间标注、尺寸标注和 CAD 输出。

### 1.3 脚本调用总原则
图形生成必须严格调用当前项目 `scripts` 目录中的实际程序，不得自行替换为不存在的脚本或绕过脚本直接猜画。图片任务必须先调用 `processing.py`，墙体任务调用 `generate_by_pic.py`，文字尺寸任务调用 `generate_by_txt.py`，窗户任务调用 `generate_windows.py`，门任务调用 `generate_doors.py`。

## 2. 强制执行流程
### 2.1 输入解析与图片预处理
图片输入必须先运行：
`python scripts/processing.py <输入图片> --output-dir <预处理输出目录>`
然后读取该目录中的 `result.json`、`overlay.png`、`masks/walls.png`、`masks/doors.png`、`masks/windows.png`、`masks/dimensions.png`、`masks/room_labels.png` 和 `masks/other.png`。OCR 不可用时，text=null 的候选不得视为确定文字。
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
生成 Wall、Opening、Door、Window、Room 和 UncertainObject 数据。用户要求优先于图片明确尺寸，图片明确尺寸优先于几何推定。
### 2.14 CAD MCP绘制
不得直接由 Skill 手工绘制图形，必须按任务类型调用对应脚本：图片墙体调用 `generate_by_pic.py`，文字尺寸调用 `generate_by_txt.py`，窗户调用 `generate_windows.py`，门调用 `generate_doors.py`。各脚本的输出必须作为下一脚本的输入。
### 2.15 绘后检查
检查双线墙、墙厚、墙体连接、门窗是否位于墙上、门窗洞口是否真实断开、房间是否闭合、尺寸比例是否正确，以及是否存在重叠和悬空。
### 2.16 局部纠错
只修改对应局部对象，并重新检查相邻墙体、门窗、房间和尺寸关系。不得因局部纠错改变无关区域。

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
包含 id、name、x、y、width、height、boundary_walls、source 和 confidence；房间应由墙体围合形成闭合空间。
### 3.6 UncertainObject
记录模糊、重叠或无法确认的墙体、门窗、文字和尺寸候选，包含 geometry、reason、confidence 和 INFERRED 状态。

## 4. 信息可信度
### 4.1 USER
用户明确提供的墙厚、房间尺寸、门窗尺寸或布局关系，优先级最高。
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
房间名称优先使用用户输入或图片明确文字，文字放在对应房间内部，不得遮挡墙体、门窗和尺寸。
### 6.6 尺寸
按可靠尺寸建立比例并绘制必要尺寸。推定尺寸必须与确定尺寸区分，尺寸线应离开建筑墙体并保持清晰间隙。

## 7. 纠错机制
采用“识别—建模—校核—局部修正—再次校核”闭环。无法可靠判断时保留候选并标记 INFERRED，必要时说明不确定区域，不得凭空制造复杂结构。

## 7.1 实际脚本调用流程

### 图片输入流程

1. 预处理：`python scripts/processing.py <source.png> --output-dir <preprocessed>`
2. 墙体生成：`python scripts/generate_by_pic.py <source.png> <walls.dxf> --data scripts/data.json --overall-width-mm <总宽> --overall-height-mm <总高> --wall-bbox <x1> <y1> <x2> <y2> --horizontal-chain-mm <横向尺寸链> --vertical-chain-top-down-mm <纵向尺寸链>`
3. 窗户生成：`python scripts/generate_windows.py <含窗图片> <walls.dxf> <walls_windows.dxf> --data scripts/data.json --preprocessing-result <preprocessed/result.json>`
4. 门生成：`python scripts/generate_doors.py <含门图片> <上一阶段图片> <walls_windows.dxf> <final.dxf> --data scripts/data.json --preprocessing-result <preprocessed/result.json>`

墙体脚本输出的 DXF 必须先校核通过，才能传给 `generate_windows.py`；窗户脚本输出必须先校核通过，才能传给 `generate_doors.py`。尺寸链和墙体外框不能凭空填写，必须来自用户、图纸标注或经过复核的图像测量。

### 文字输入流程

文字尺寸任务必须调用：
`python scripts/generate_by_txt.py --config <配置.json>`
或使用其交互入口：
`python scripts/generate_by_txt.py --interactive`

`generate_by_txt.py` 直接连接当前活动 CAD 文档；执行前必须确认目标 CAD 已启动并存在活动图形文档。

### 脚本职责边界

- `processing.py` 只负责图片预处理和识别证据输出。
- `generate_by_pic.py` 只负责图片墙体及墙体尺寸 DXF。
- `generate_by_txt.py` 负责文字配置到活动 CAD 的生成。
- `generate_windows.py` 只负责在已校核墙体 DXF 上切窗洞并生成窗框。
- `generate_doors.py` 只负责在已校核墙体和窗户 DXF 上切门洞并生成门图形。

不得使用不存在的 `generate.py`，不得让 `generate_by_pic.py` 代替门窗脚本，也不得让门窗脚本跳过前置墙体结果。

## 8. 输入方式与使用示例
图片输入：请根据这张户型图生成墙体、窗户和门，墙体使用双线墙，并添加必要尺寸标注。必须按 processing.py → generate_by_pic.py → generate_windows.py → generate_doors.py 顺序执行。
文字输入：生成一个两室一厅户型图；客厅 6000×4500；主卧 4500×3600；次卧 3600×3300；外墙厚度 240；内墙厚度 200。使用 generate_by_txt.py 的配置入口。

## 9. 默认参数
单位为 mm；外墙厚度根据图片或用户要求确定；内墙厚度根据图片或用户要求确定；普通卧室门 900；卫生间门 800；入户门 1000～1200；普通窗宽 1200～1800。可靠图片或用户数据优先于默认值。

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
