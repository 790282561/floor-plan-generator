# 户型图生成器 - 详细使用指南

## 目录
1. [快速开始](#快速开始)
2. [命令行使用](#命令行使用)
3. [配置文件格式](#配置文件格式)
4. [Python API](#python-api)
5. [常见问题](#常见问题)

---

## 快速开始

### 基本使用

直接告诉AI助手您的需求：

```
生成一个两室一厅户型图
客厅：6000×4500
主卧：4500×3600
次卧：3600×3300
厨房：3000×2400
卫生间：2400×1800
```

AI助手会自动调用技能生成CAD文件。

---

## 命令行使用

### 1. 使用默认配置生成

```bash
python "C:\Users\Administrator\.codebuddy\skills\floor-plan-generator\scripts\generate.py"
```

### 2. 使用配置文件生成

```bash
python "C:\Users\Administrator\.codebuddy\skills\floor-plan-generator\scripts\generate.py" --config my_config.json
```

### 3. 交互式生成

```bash
python "C:\Users\Administrator\.codebuddy\skills\floor-plan-generator\scripts\generate.py" --interactive
```

### 4. 指定输出文件名

```bash
python "C:\Users\Administrator\.codebuddy\skills\floor-plan-generator\scripts\generate.py" --output "我的户型图.dwg"
```

---

## 配置文件格式

配置文件为JSON格式，示例：

```json
{
  "rooms": [
    {
      "name": "客厅",
      "width": 6000,
      "height": 4500
    },
    {
      "name": "主卧",
      "width": 4500,
      "height": 3600
    }
  ],
  "wall_thickness": 240,
  "door_width": 900,
  "bathroom_door_width": 700,
  "window_width": 1200,
  "output_filename": "户型图",
  "output_dir": "C:\\Users\\Administrator\\Desktop"
}
```

### 配置参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| rooms | Array | 是 | 房间列表 |
| rooms[].name | String | 是 | 房间名称 |
| rooms[].width | Integer | 是 | 房间宽度(mm) |
| rooms[].height | Integer | 是 | 房间高度(mm) |
| wall_thickness | Integer | 否 | 墙厚，默认240mm |
| door_width | Integer | 否 | 门宽，默认900mm |
| bathroom_door_width | Integer | 否 | 卫生间门宽，默认700mm |
| window_width | Integer | 否 | 窗户宽，默认1200mm |
| output_filename | String | 否 | 输出文件名，默认"户型图" |
| output_dir | String | 否 | 输出目录，默认桌面 |

---

## Python API

### 基本用法

```python
from generate import FloorPlanGenerator, FloorPlanConfig

# 创建配置
config = FloorPlanConfig()
config.rooms = [
    {"name": "客厅", "width": 6000, "height": 4500},
    {"name": "主卧", "width": 4500, "height": 3600},
]
config.output_filename = "我的户型图"

# 生成
generator = FloorPlanGenerator(config)
output_path = generator.generate()
```

### 从配置文件加载

```python
config = FloorPlanConfig.from_json("config.json")
generator = FloorPlanGenerator(config)
generator.generate()
```

### 保存配置

```python
config = FloorPlanConfig()
config.to_json("my_config.json")
```

---

## 常见户型示例

### 一室一厅

```json
{
  "rooms": [
    {"name": "客厅", "width": 4500, "height": 4000},
    {"name": "卧室", "width": 4000, "height": 3500},
    {"name": "厨房", "width": 2500, "height": 2000},
    {"name": "卫生间", "width": 2000, "height": 1800}
  ]
}
```

### 两室一厅

```json
{
  "rooms": [
    {"name": "客厅", "width": 6000, "height": 4500},
    {"name": "主卧", "width": 4500, "height": 3600},
    {"name": "次卧", "width": 3600, "height": 3300},
    {"name": "厨房", "width": 3000, "height": 2400},
    {"name": "卫生间", "width": 2400, "height": 1800}
  ]
}
```

### 三室两厅

```json
{
  "rooms": [
    {"name": "客厅", "width": 6000, "height": 4500},
    {"name": "餐厅", "width": 3600, "height": 3000},
    {"name": "主卧", "width": 4500, "height": 3600},
    {"name": "次卧1", "width": 3600, "height": 3300},
    {"name": "次卧2", "width": 3300, "height": 3000},
    {"name": "厨房", "width": 3000, "height": 2400},
    {"name": "卫生间", "width": 2400, "height": 1800}
  ]
}
```

---

## 常见问题

### Q: 尺寸单位是什么？
A: 所有尺寸单位都是毫米(mm)。

### Q: 生成的DWG文件用什么软件打开？
A: 可以使用AutoCAD、QCAD、DraftSight、LibreCAD等CAD软件打开。

### Q: 如何修改生成后的图纸？
A: 直接用CAD软件打开DWG文件即可编辑，所有元素都在对应的图层上。

### Q: 可以生成其他格式吗？
A: 目前支持DWG格式。DXF格式可以通过CAD软件导出。

### Q: 如何调整房间布局？
A: 目前版本自动布局。未来版本将支持手动指定房间位置。

### Q: 支持多层建筑吗？
A: 目前版本仅支持单层户型图。

---

## 技术支持

如遇到问题，请检查：
1. Python版本是否 >= 3.8
2. ezdxf库是否正确安装
3. 输出目录是否存在且可写

安装依赖：
```bash
pip install ezdxf
```
