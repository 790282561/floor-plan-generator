# 户型图图例说明

- `wall_legend.jpg`：墙体涂实。
- `door_legend.jpg`：平开门，圆弧表示开启方向。
- `window_legend.jpg`：普通窗，两条平行线表示窗框。
- `sliding_door_legend.jpg`：推拉门，两扇门板相互搭接。
- `wall_corner_legend.jpg`：墙角连接示例。
- `window_corner_legend.jpg`：转角窗示例；水平窗段与竖直窗段在同一墙角连续相接，形成 L 形多线窗框。
- `annotated_floorplan_example.jpg`：完整户型图中的统一标记示例。
- `annotated_floorplan_example2.jpg`：新增综合示例；红框标出了左上、左下和右上的转角窗，并同时展示门弧与普通墙线的分类边界。

分类优先级：检测到四分之一圆门弧及铰链门扇线时，该区域必须分类为门，不得同时生成窗。普通窗必须是同一墙段上的四条平行线；转角窗必须是四条连续 L 形线，每条都由水平段和竖直段在同一墙角相接。未检测到完整四线证据时，不得分类为窗或转角窗。
