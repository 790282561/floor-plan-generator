"""
户型图自动生成器 - 核心脚本
根据房间尺寸参数直接在当前 CAD 活跃文档中生成户型线稿

使用方法:
1. 命令行: python generate.py --config config.json
2. 交互式: python generate.py --interactive
3. Python API: from generate import FloorPlanGenerator, FloorPlanConfig

要求: 运行前已打开 AutoCAD (或兼容 CAD) 并有一个图形文档。
脚本不生成 DXF/DWG 文件，所有图元直接绘制到当前活跃文档的模型空间。
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, List

try:
    import pythoncom
    import win32com.client
    from win32com.client import VARIANT
except ImportError:
    print("[错误] 缺少 pywin32，请先安装: pip install pywin32")
    sys.exit(1)

# 常见 CAD 的 COM ProgID，依次尝试连接
CAD_PROG_IDS = ["AutoCAD.Application", "Gcad.Application", "ZWCAD.Application"]

# 文字对齐常量 (acAlignmentMiddleCenter)
AC_ALIGNMENT_MIDDLE_CENTER = 10


def acad_pt(x, y, z=0.0):
    """构造 AutoCAD COM 三维点 (VT_ARRAY | VT_R8)"""
    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(x), float(y), float(z)))


def acad_2d(coords):
    """构造 LightWeightPolyline 扁平顶点数组 [(x1,y1), (x2,y2), ...] -> [x1,y1,x2,y2,...]"""
    flat = []
    for x, y in coords:
        flat.extend([float(x), float(y)])
    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, flat)


class FloorPlanConfig:
    """户型图配置类"""

    def __init__(self):
        # 默认两室一厅配置
        self.rooms: List[Dict] = [
            {"name": "客厅", "width": 6000, "height": 4500},
            {"name": "主卧", "width": 4500, "height": 3600},
            {"name": "次卧", "width": 3600, "height": 3300},
            {"name": "厨房", "width": 3000, "height": 2400},
            {"name": "卫生间", "width": 2400, "height": 1800},
        ]
        self.wall_thickness = 240
        self.door_width = 900
        self.bathroom_door_width = 700
        self.window_width = 1200
        # 以下两个字段仅为兼容旧配置文件保留，直绘模式下不使用
        self.output_filename = "户型图"
        self.output_dir = os.path.expanduser("~\\Desktop")

    @classmethod
    def from_json(cls, json_path: str) -> 'FloorPlanConfig':
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = cls()
        if 'rooms' in data:
            config.rooms = data['rooms']
        if 'wall_thickness' in data:
            config.wall_thickness = data['wall_thickness']
        if 'door_width' in data:
            config.door_width = data['door_width']
        if 'bathroom_door_width' in data:
            config.bathroom_door_width = data['bathroom_door_width']
        if 'window_width' in data:
            config.window_width = data['window_width']
        if 'output_filename' in data:
            config.output_filename = data['output_filename']
        if 'output_dir' in data:
            config.output_dir = data['output_dir']

        return config

    def to_json(self, json_path: str):
        """保存配置到JSON文件"""
        data = {
            'rooms': self.rooms,
            'wall_thickness': self.wall_thickness,
            'door_width': self.door_width,
            'bathroom_door_width': self.bathroom_door_width,
            'window_width': self.window_width,
            'output_filename': self.output_filename,
            'output_dir': self.output_dir
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class FloorPlanGenerator:
    """户型图生成器 - 直接绘制到当前 CAD 活跃文档"""

    def __init__(self, config: FloorPlanConfig):
        self.config = config
        self.acad = None
        self.doc = None
        self.msp = None
        self.layout = {}

        self._connect_acad()
        self._setup_layers()
        self.text_style = self._setup_text_style()

    # ------------------------------------------------------------------
    # CAD 连接与图层
    # ------------------------------------------------------------------
    def _connect_acad(self):
        """连接正在运行的 CAD 程序，获取当前活跃文档"""
        errors = []
        for prog_id in CAD_PROG_IDS:
            try:
                self.acad = win32com.client.GetActiveObject(prog_id)
                break
            except Exception as e:
                errors.append(f"{prog_id}: {e}")

        if self.acad is None:
            raise RuntimeError(
                "未找到正在运行的 CAD 程序，请先打开 AutoCAD (或兼容 CAD) "
                "并确认有一个图形文档。\n" + "\n".join(errors)
            )

        try:
            self.doc = self.acad.ActiveDocument
        except Exception as e:
            raise RuntimeError(f"无法获取 CAD 活跃文档: {e}")

        if self.doc is None:
            raise RuntimeError("CAD 中没有活跃文档，请新建或打开一个图形文档。")

        self.msp = self.doc.ModelSpace
        print(f"[*] 已连接 CAD ({self.acad.Name})，目标文档: {self.doc.Name}")

    def _setup_layers(self):
        """在当前文档创建标准图层（已存在则复用）"""
        # (名称, ACI颜色, 线型, 线重/100mm)
        layer_defs = [
            ("WALLS", 7, "Continuous", 50),
            ("INNER_WALLS", 7, "Continuous", 35),
            ("DOORS", 3, "Continuous", 25),
            ("WINDOWS", 5, "Continuous", 25),
            ("DIMENSIONS", 1, "Continuous", 20),
            ("ROOM_NAMES", 3, "Continuous", 50),
            ("TEXT", 7, "Continuous", 30),
        ]

        for name, color, linetype, lineweight in layer_defs:
            try:
                layer = self.doc.Layers.Item(name)
            except Exception:
                layer = self.doc.Layers.Add(name)
            try:
                layer.color = color
            except Exception:
                pass
            try:
                layer.Linetype = linetype
            except Exception:
                pass
            try:
                layer.Lineweight = lineweight
            except Exception:
                pass

    def _setup_text_style(self) -> str:
        """确保存在支持中文的文字样式，杜绝文字显示为 ???

        优先级:
        1. 国标 SHX: gbenor.shx + gbcbig.shx 大字体 (设计院常规做法)
        2. TrueType: simsun.ttc (宋体, 中文 Windows 必备)
        3. 其他常见中文字体依次回退
        """
        style_name = "FP-TEXT"
        try:
            ts = self.doc.TextStyles.Item(style_name)
        except Exception:
            ts = self.doc.TextStyles.Add(style_name)

        # 已配置过且含大字体/中文字体则直接复用
        try:
            if getattr(ts, "BigFontFile", "") or "sim" in str(getattr(ts, "fontFile", "")).lower() \
                    or "gb" in str(getattr(ts, "fontFile", "")).lower():
                return style_name
        except Exception:
            pass

        # 1) 国标 SHX + 中文大字体
        try:
            cad_fonts = os.path.join(self.acad.Path, "Fonts")
            if os.path.isfile(os.path.join(cad_fonts, "gbenor.shx")) and \
               os.path.isfile(os.path.join(cad_fonts, "gbcbig.shx")):
                ts.fontFile = "gbenor.shx"
                ts.BigFontFile = "gbcbig.shx"
                print("    - 文字样式: gbenor.shx + gbcbig.shx (国标大字体)")
                return style_name
        except Exception:
            pass

        # 2/3) TrueType 中文字体回退
        win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        for font in ("simsun.ttc", "simsun.ttf", "msyh.ttc", "simhei.ttf", "simfang.ttf", "stkaiti.ttf"):
            if not os.path.isfile(os.path.join(win_fonts, font)):
                continue
            try:
                ts.fontFile = font
                print(f"    - 文字样式: {font} (TrueType)")
                return style_name
            except Exception:
                continue

        print("[警告] 未找到可用的中文字体，文字可能显示为 ???，请手动安装中文字体后重试")
        return style_name

    # ------------------------------------------------------------------
    # 绘图基础封装
    # ------------------------------------------------------------------
    def _add_lwpolyline(self, coords, layer, close=False):
        """绘制多段线"""
        coords = list(coords)
        if close and coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        poly = self.msp.AddLightWeightPolyline(acad_2d(coords))
        poly.Closed = close
        poly.Layer = layer
        return poly

    def _add_line(self, p1, p2, layer):
        """绘制直线"""
        line = self.msp.AddLine(acad_pt(*p1), acad_pt(*p2))
        line.Layer = layer
        return line

    def _add_arc(self, center, radius, start_deg, end_deg, layer):
        """绘制圆弧（角度使用度，内部转弧度）"""
        arc = self.msp.AddArc(
            acad_pt(*center), float(radius),
            math.radians(start_deg), math.radians(end_deg)
        )
        arc.Layer = layer
        return arc

    def _add_text(self, text, pos, height, layer):
        """绘制居中对齐文字（强制使用支持中文的样式，避免 ??? 乱码）"""
        t = self.msp.AddText(text, acad_pt(*pos), float(height))
        try:
            t.StyleName = self.text_style
        except Exception:
            pass
        t.Layer = layer
        try:
            t.Alignment = AC_ALIGNMENT_MIDDLE_CENTER
            t.TextAlignmentPoint = acad_pt(*pos)
        except Exception:
            pass
        return t

    # ------------------------------------------------------------------
    # 布局设计
    # ------------------------------------------------------------------
    def _design_layout(self) -> Dict:
        """设计户型布局"""
        wall = self.config.wall_thickness
        rooms = self.config.rooms

        if len(rooms) == 2:
            # 一室一厅
            layout = self._layout_1br(rooms, wall)
        elif len(rooms) == 3:
            # 两室一厅
            layout = self._layout_2br(rooms, wall)
        elif len(rooms) == 4:
            # 三室一厅
            layout = self._layout_3br(rooms, wall)
        elif len(rooms) == 5:
            # 三室两厅
            layout = self._layout_3br2l(rooms, wall)
        else:
            # 默认布局
            layout = self._layout_default(rooms, wall)

        return layout

    def _layout_1br(self, rooms, wall) -> Dict:
        """一室一厅布局"""
        layout = {}
        living = rooms[0] if "客厅" in rooms[0]["name"] else rooms[1]
        bedroom = rooms[0] if "卧" in rooms[0]["name"] else rooms[1]

        x, y = wall, wall
        layout["living_room"] = {"x": x, "y": y, **living}
        layout["bedroom"] = {"x": x, "y": y + living["height"], **bedroom}

        return layout

    def _layout_2br(self, rooms, wall) -> Dict:
        """两室一厅布局"""
        layout = {}
        # 假设: 客厅、厨房、两个卧室
        living = next((r for r in rooms if "客厅" in r["name"]), rooms[0])
        master = next((r for r in rooms if "主卧" in r["name"]), rooms[1])
        second = next((r for r in rooms if "次卧" in r["name"]), None)
        kitchen = next((r for r in rooms if "厨房" in r["name"]), None)
        bathroom = next((r for r in rooms if "卫生" in r["name"]), None)

        x, y = wall, wall
        # 客厅在左下
        layout["living_room"] = {"x": x, "y": y, **living}

        # 主卧在客厅上方
        layout["master_bedroom"] = {
            "x": x + living["width"],
            "y": y + living["height"] - master["height"],
            **master
        }

        # 次卧在客厅右侧
        if second:
            layout["second_bedroom"] = {
                "x": x,
                "y": y + living["height"],
                **second
            }

        # 厨房
        if kitchen:
            layout["kitchen"] = {
                "x": x + second["width"] if second else x,
                "y": y - kitchen["height"],
                **kitchen
            }

        # 卫生间
        if bathroom:
            layout["bathroom"] = {
                "x": x + living["width"],
                "y": y - bathroom["height"],
                **bathroom
            }

        return layout

    def _layout_3br(self, rooms, wall) -> Dict:
        """三室一厅布局"""
        return self._layout_2br(rooms, wall)

    def _layout_3br2l(self, rooms, wall) -> Dict:
        """三室两厅布局"""
        layout = {}
        rooms_copy = rooms.copy()

        # 找出所有房间类型
        living = next((r for r in rooms_copy if "客厅" in r["name"]), {"name": "客厅", "width": 6000, "height": 4500})
        dining = next((r for r in rooms_copy if "餐厅" in r["name"]), {"name": "餐厅", "width": 3000, "height": 3000})
        master = next((r for r in rooms_copy if "主卧" in r["name"]), {"name": "主卧", "width": 4500, "height": 3600})
        second = next((r for r in rooms_copy if "次卧" in r["name"] and "主" not in r["name"]), {"name": "次卧", "width": 3600, "height": 3300})
        kitchen = next((r for r in rooms_copy if "厨房" in r["name"]), {"name": "厨房", "width": 3000, "height": 2400})
        bathroom = next((r for r in rooms_copy if "卫生" in r["name"]), {"name": "卫生间", "width": 2400, "height": 1800})

        x, y = wall, wall

        # 客厅
        layout["living_room"] = {"x": x, "y": y, **living}

        # 餐厅在客厅上方
        layout["dining_room"] = {
            "x": x,
            "y": y + living["height"],
            **dining
        }

        # 主卧
        layout["master_bedroom"] = {
            "x": x + living["width"],
            "y": y + dining["height"] - master["height"],
            **master
        }

        # 次卧
        layout["second_bedroom"] = {
            "x": x + living["width"],
            "y": y,
            **second
        }

        # 厨房
        layout["kitchen"] = {
            "x": x - kitchen["width"],
            "y": y,
            **kitchen
        }

        # 卫生间
        layout["bathroom"] = {
            "x": x - bathroom["width"],
            "y": y + kitchen["height"],
            **bathroom
        }

        return layout

    def _layout_default(self, rooms, wall) -> Dict:
        """默认布局 - 根据房间数量智能分配"""
        layout = {}
        x, y = wall, wall

        # 客厅优先
        living = next((r for r in rooms if "客厅" in r["name"] or "起居" in r["name"]), rooms[0])
        layout["room_0"] = {"x": x, "y": y, **living}

        # 依次排列其他房间
        for i, room in enumerate(rooms[1:], 1):
            prev_room = rooms[i-1]
            layout[f"room_{i}"] = {
                "x": x + prev_room["width"],
                "y": y,
                **room
            }

        return layout

    # ------------------------------------------------------------------
    # 图元绘制
    # ------------------------------------------------------------------
    def draw_outer_walls(self):
        """绘制外墙（双线）"""
        wall = self.config.wall_thickness

        all_rooms = list(self.layout.values())
        min_x = min(r["x"] for r in all_rooms) - wall
        min_y = min(r["y"] for r in all_rooms) - wall
        max_x = max(r["x"] + r["width"] for r in all_rooms) + wall
        max_y = max(r["y"] + r["height"] for r in all_rooms) + wall

        # 外边
        self._add_lwpolyline(
            [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
            layer="WALLS", close=True
        )

        # 内边
        self._add_lwpolyline(
            [(min_x + wall, min_y + wall),
             (max_x - wall, min_y + wall),
             (max_x - wall, max_y - wall),
             (min_x + wall, max_y - wall)],
            layer="WALLS", close=True
        )

    def draw_inner_walls(self):
        """绘制内墙"""
        wall = self.config.wall_thickness

        for name, room in self.layout.items():
            # 为每个房间绘制分隔墙
            x, y = room["x"], room["y"]
            w, h = room["width"], room["height"]

            # 上边
            self._add_lwpolyline(
                [(x, y + h - wall), (x + w, y + h - wall)],
                layer="INNER_WALLS"
            )
            # 右边
            self._add_lwpolyline(
                [(x + w - wall, y), (x + w - wall, y + h)],
                layer="INNER_WALLS"
            )

    def draw_doors(self):
        """绘制门"""
        door_width = self.config.door_width

        for name, room in self.layout.items():
            center_x = room["x"] + room["width"] / 2
            center_y = room["y"] + room["height"] / 2

            # 绘制门扇 (简单的圆弧表示)
            if door_width <= room["width"]:
                # 水平门
                self._add_arc(
                    center=(center_x, room["y"]),
                    radius=door_width / 2,
                    start_deg=90, end_deg=180,
                    layer="DOORS"
                )
            if door_width <= room["height"]:
                # 垂直门
                self._add_arc(
                    center=(room["x"] + room["width"], center_y),
                    radius=door_width / 2,
                    start_deg=0, end_deg=90,
                    layer="DOORS"
                )

    def draw_windows(self):
        """绘制窗户"""
        wall = self.config.wall_thickness
        window_width = self.config.window_width
        half_wall = wall / 2

        for name, room in self.layout.items():
            center_x = room["x"] + room["width"] / 2
            center_y = room["y"] + room["height"] / 2

            # 窗户中线
            if window_width <= room["width"]:
                # 水平窗户
                win_x = center_x - window_width / 2
                self._add_line(
                    (win_x, room["y"] + room["height"] - half_wall),
                    (win_x + window_width, room["y"] + room["height"] - half_wall),
                    layer="WINDOWS"
                )
            if window_width <= room["height"]:
                # 垂直窗户
                win_y = center_y - window_width / 2
                self._add_line(
                    (room["x"] - half_wall, win_y),
                    (room["x"] - half_wall, win_y + window_width),
                    layer="WINDOWS"
                )

    def add_dimensions(self):
        """添加尺寸标注"""
        wall = self.config.wall_thickness
        offset = 800

        all_rooms = list(self.layout.values())
        min_x = min(r["x"] for r in all_rooms) - wall
        min_y = min(r["y"] for r in all_rooms) - wall
        max_x = max(r["x"] + r["width"] for r in all_rooms) + wall
        max_y = max(r["y"] + r["height"] for r in all_rooms) + wall

        # 总尺寸
        self._add_text(
            f"{max_x - min_x}",
            pos=((min_x + max_x) / 2, min_y - offset + 100),
            height=250, layer="DIMENSIONS"
        )

        self._add_text(
            f"{max_y - min_y}",
            pos=(min_x - offset + 100, (min_y + max_y) / 2),
            height=250, layer="DIMENSIONS"
        )

        # 尺寸线
        self._add_line((min_x, min_y - offset), (max_x, min_y - offset), layer="DIMENSIONS")
        self._add_line((min_x - offset, min_y), (min_x - offset, max_y), layer="DIMENSIONS")

    def add_room_labels(self):
        """添加房间标签"""
        for name, room in self.layout.items():
            center_x = room["x"] + room["width"] / 2
            center_y = room["y"] + room["height"] / 2

            # 房间名称
            self._add_text(
                room["name"],
                pos=(center_x, center_y),
                height=400, layer="ROOM_NAMES"
            )

            # 面积
            area = (room["width"] * room["height"]) / 1000000
            self._add_text(
                f"{area:.1f}m2",
                pos=(center_x, center_y - 500),
                height=250, layer="TEXT"
            )

    def add_title(self):
        """添加标题"""
        all_rooms = list(self.layout.values())
        max_x = max(r["x"] + r["width"] for r in all_rooms)
        max_y = max(r["y"] + r["height"] for r in all_rooms)

        total_area = sum(r["width"] * r["height"] for r in all_rooms) / 1000000

        self._add_text(
            f"{len(self.config.rooms)}室户型图",
            pos=(max_x / 2, max_y + 1500),
            height=600, layer="ROOM_NAMES"
        )

        self._add_text(
            f"总面积: {total_area:.1f}m2 | 1:100",
            pos=(max_x / 2, max_y + 1000),
            height=300, layer="TEXT"
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def generate(self) -> str:
        """在当前 CAD 活跃文档中生成户型线稿（不保存文件）"""
        print("[*] 开始生成户型图...")

        # 设计布局
        self.layout = self._design_layout()

        # 绘制各部分
        print("    - 绘制外墙...")
        self.draw_outer_walls()

        print("    - 绘制内墙...")
        self.draw_inner_walls()

        print("    - 绘制门...")
        self.draw_doors()

        print("    - 绘制窗户...")
        self.draw_windows()

        print("    - 添加标注...")
        self.add_dimensions()

        print("    - 添加房间标签...")
        self.add_room_labels()

        print("    - 添加标题...")
        self.add_title()

        # 刷新视图（不保存文件）
        try:
            self.acad.ZoomExtents()
        except Exception:
            pass
        try:
            self.doc.Regen(1)  # acActiveViewport
        except Exception:
            pass

        print(f"[OK] 户型图已绘制到当前活跃文档: {self.doc.Name}")

        # 统计
        total_area = sum(r["width"] * r["height"] for r in self.layout.values()) / 1000000
        print(f"\n[*] 户型统计:")
        print(f"    总面积: {total_area:.1f} m2")
        print(f"    房间数量: {len(self.layout)}")
        for name, room in self.layout.items():
            area = room["width"] * room["height"] / 1000000
            print(f"    - {room['name']}: {room['width']}x{room['height']} ({area:.1f}m2)")

        return self.doc.Name


def interactive_mode():
    """交互式输入模式"""
    print("\n" + "=" * 60)
    print("户型图生成器 - 交互式模式 (直接绘制到当前 CAD 活跃文档)")
    print("=" * 60)

    config = FloorPlanConfig()

    # 输入房间数量
    while True:
        try:
            num_rooms = int(input("\n请输入房间数量 (2-6): "))
            if 2 <= num_rooms <= 6:
                break
            print("请输入2-6之间的数字")
        except ValueError:
            print("请输入有效的数字")

    config.rooms = []
    room_types = ["客厅", "主卧", "次卧", "厨房", "卫生间", "餐厅", "书房", "阳台"]

    for i in range(num_rooms):
        print(f"\n--- 房间 {i+1} ---")
        name = input(f"房间名称 ({', '.join(room_types[:num_rooms])}): ").strip()
        if not name:
            name = room_types[i] if i < len(room_types) else f"房间{i+1}"

        while True:
            try:
                width = int(input("宽度 (mm): "))
                height = int(input("高度 (mm): "))
                if width > 0 and height > 0:
                    break
                print("请输入正数")
            except ValueError:
                print("请输入有效的数字")

        config.rooms.append({"name": name, "width": width, "height": height})

    # 其他参数
    print("\n--- 其他参数 (直接回车使用默认值) ---")

    wall = input(f"墙厚 (mm) [默认240]: ").strip()
    if wall:
        config.wall_thickness = int(wall)

    door = input(f"门宽 (mm) [默认900]: ").strip()
    if door:
        config.door_width = int(door)

    window = input(f"窗宽 (mm) [默认1200]: ").strip()
    if window:
        config.window_width = int(window)

    # 生成
    generator = FloorPlanGenerator(config)
    doc_name = generator.generate()

    print(f"\n[DONE] 线稿已绘制到当前 CAD 文档: {doc_name}")
    return doc_name


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="户型图自动生成器 (直绘当前 CAD 活跃文档)")
    parser.add_argument("--config", "-c", help="配置文件路径 (JSON)")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式输入模式")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.config:
        # 从配置文件加载
        config = FloorPlanConfig.from_json(args.config)
        generator = FloorPlanGenerator(config)
        generator.generate()
    else:
        # 使用默认配置
        print("[*] 使用默认配置生成户型图...")
        config = FloorPlanConfig()
        generator = FloorPlanGenerator(config)
        generator.generate()


if __name__ == "__main__":
    main()
