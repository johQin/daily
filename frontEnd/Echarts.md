# Echarts

---

# 1.API

## 1.1 echarts

全局 echarts 对象

1. echarts.**init**( dom, theme? , opts?  ) ——返回echartsinstance
2. echarts.**connect**( group )—— 多个图表示例实现联动
3. echarts.**disconnect**( group )——解除图表实例的联动
4. echarts.**dispose**()——销毁实例
5. echarts.**getInstanceByDom**()——获取 dom 容器上的实例。
6. echarts.**registerMap**()——注册可用的地图
7. echarts.**getMap**()——获取已注册的地图
8. echarts.**registerTheme**()——注册主题

## 1.2 echartsInstance

通过 [echarts.init](https://www.echartsjs.com/zh/api.html#echarts.init) 创建的实例。

1. echartsInstance.**group**——图表的分组，用于[联动](https://www.echartsjs.com/zh/api.html#echarts.connect)
2. echartsInstance.**setOption**([option](https://www.echartsjs.com/zh/option.html), notMerge, lazyUpdate)——设置图表实例的配置项以及数据，万能接口，所有参数和数据的修改都可以通过 `setOption` 完成，option具体见配置项手册。
3. echartsInstance.**getWidth**()和**getHeight**()——获取实例的宽高
4. echartsInstance.**getDom**()——获取实例的dom节点
5. echartsInstance.**getOption**()——获取当前实例中维护的 `option` 对象
6. echartsInstance.**resize**()——改变或获取图表尺寸
7. echartsInstance.**dispatchAction**()——触发图表行为，更多见 [action](https://www.echartsjs.com/zh/api.html#action) 和 [events](https://www.echartsjs.com/zh/api.html#events) 的文档。
8. echartsInstance.**on**(eventName,query,handler,context?)——绑定事件处理函数，ECharts 中的事件有两种，一种是鼠标事件，在鼠标点击某个图形上会触发，还有一种是 调用 [dispatchAction](https://www.echartsjs.com/zh/api.html#echartsInstance.dispatchAction) 后触发的事件。每个 action 都会有对应的事件，具体见 [action](https://www.echartsjs.com/zh/api.html#action) 和 [events](https://www.echartsjs.com/zh/api.html#events) 的文档。<span style='color:red'>事件名，包括鼠标事件名和action名</span>
9. echartsInstance.**off**()——解绑事件处理函数
10. echartsInstance.**convertToPixel**()——转换坐标系上的点到像素坐标值。
11. echartsInstance.**convertFromPixel**()——转换像素坐标值到逻辑坐标系上的点。是 [convertToPixel](https://www.echartsjs.com/zh/api.html#echartsInstance.convertToPixel) 的逆运算。
12. echartsInstance.**showLoading**()和**hideLoading**()——显示(隐藏)加载动画效果。
13. echartsInstance.**getDataURL**()和**getConnectedDataURL**()——导出图表(联动图表)图片，返回一个 base64 的 URL，
14. echartsInstance.**appendData**()——分片加载数据和增量渲染
15. echartsInstance.**clear**——清空当前实例，会移除实例中所有的组件和图表。
16. echartsInstance.**isDisposed**——判断当前实例是否被释放
17. echartsInstance.**dispose**——销毁实例

## 1.3 action

ECharts 中支持的图表行为，通过 [dispatchAction](https://www.echartsjs.com/zh/api.html#echartsInstance.dispatchAction) 触发。

1. highlight  |  downplay——高亮（取消高亮）指定的数据图形

2. **legend**——图例legend相关行为，必须引入[图例组件](https://www.echartsjs.com/zh/option.html#legend)后才能使用。

   折叠详情

3. **tooltip** ——[提示框组件](https://www.echartsjs.com/zh/option.html#tooltip)相关的行为，必须引入[提示框组件](https://www.echartsjs.com/zh/option.html#tooltip)后才能使用。

4. **dataZoom** ——[数据区域缩放组件](https://www.echartsjs.com/zh/option.html#dataZoom)相关的行为，必须引入[数据区域缩放组件](https://www.echartsjs.com/zh/option.html#dataZoom)后才能使用。

5. **visualMap**——选取映射的数值范围。

6. **timeline**——[时间轴组件](https://www.echartsjs.com/zh/option.html#timeline)相关的行为，必须引入[时间轴组件](https://www.echartsjs.com/zh/option.html#timeline)后才能使用。

7. **toolbox**——[工具栏组件](https://www.echartsjs.com/zh/option.html#toolbox)相关的行为，必须引入[工具栏组件](https://www.echartsjs.com/zh/option.html#toolbox)后才能使用。

8. **pie**——[饼图](https://www.echartsjs.com/zh/option.html#series-pie)相关的行为，必须引入[饼图](https://www.echartsjs.com/zh/option.html#series-pie)后才能使用。

9. **geo**——[地图组件](https://www.echartsjs.com/zh/option.html#series-geo)相关的行为，必须引入[地图组件](https://www.echartsjs.com/zh/option.html#geo)后才能使用。

10. **map**——[地图图表](https://www.echartsjs.com/zh/option.html#series-map)相关的行为，必须引入[地图图表](https://www.echartsjs.com/zh/option.html#series-map)后才能使用。

11. **graph**——[关系图](https://www.echartsjs.com/zh/option.html#series-graph) 相关的行为，必须引入 [关系图](https://www.echartsjs.com/zh/option.html#series-graph) 后才能使用。

12. **brush**——[区域选择](https://www.echartsjs.com/zh/option.html#brush)相关的行为。

    

## 1.4 events

在 ECharts 中主要通过 [on](https://www.echartsjs.com/zh/api.html#echartsInstance.on) 方法添加事件处理函数，该文档描述了所有 ECharts 的事件列表。

ECharts 中的事件分为两种，一种是鼠标事件，在鼠标点击某个图形上会触发，还有一种是 调用 [dispatchAction](https://www.echartsjs.com/zh/api.html#echartsInstance.dispatchAction) 后触发的事件。

### 1.4.1 鼠标事件

鼠标事件的事件参数是事件对象的数据的各个属性，具体见各个图表类型的 label formatter 回调函数的 `params`。

鼠标事件包括 `'click'`、`'dblclick'`、`'mousedown'`、`'mousemove'`、`'mouseup'`、`'mouseover'`、`'mouseout'`、`'globalout'`、`'contextmenu'`。

### 1.4.2 dispatchAction触发的事件

.................



# 2. 配置项

setOption(  option  ) —— option为配置项

1. title——标题组件，包含主标题和副标题。
2. legend——图例组件，图例组件展现了不同系列的标记(symbol)，颜色和名字。可以通过点击图例控制哪些系列不显示。
3. grid——将整个画布分为几个部分，而不是认为的网格线。
4. xAxis——直角坐标系 grid 中的 x 轴
5. yAxis——直角坐标系 grid 中的 y 轴
6. polar——极坐标系，每个极坐标系拥有一个[角度轴](https://www.echartsjs.com/zh/option.html#angleAxis)和一个[半径轴](https://www.echartsjs.com/zh/option.html#radiusAxis)。
7. radiusAxis——极坐标系的径向轴。
8. angleAxis——极坐标系的角度轴。
9. radar——雷达图坐标系组件，只适用于[雷达图](https://www.echartsjs.com/zh/option.html#series-radar)。
10. dataZoom——`dataZoom` 组件 用于区域缩放，inside-坐标系内平移缩放，slider-滑动条型数据区域缩放组件
11. visualMap——`visualMap` 是视觉映射组件，用于进行『视觉编码』，也就是将数据映射到视觉元素（视觉通道）。视觉元素可以是图形的类别，图元的大小，颜色透明度等等。分continuous—连续型视觉映射组件和Piecewise—分段型视觉映射组件
12. tooltip——提示框组件。可以设置在多种地方（全局，坐标系，系列和data[i]）
13. axisPointer——坐标轴指示器是指示坐标轴当前刻度的工具。
14. toolbox——工具栏。内置有[导出图片](https://www.echartsjs.com/zh/option.html#toolbox.feature.saveAsImage)，[数据视图](https://www.echartsjs.com/zh/option.html#toolbox.feature.dataView)，[动态类型切换](https://www.echartsjs.com/zh/option.html#toolbox.feature.magicType)，[数据区域缩放](https://www.echartsjs.com/zh/option.html#toolbox.feature.dataZoom)，[重置](https://www.echartsjs.com/zh/option.html#toolbox.feature.reset)五个工具。
15. brush——`brush` 是区域选择组件，用户可以选择图中一部分数据，从而便于向用户展示被选中数据，或者他们的一些统计计算结果。
16. geo——地理坐标系组件用于地图的绘制，支持在地理坐标系上绘制[散点图](https://www.echartsjs.com/zh/option.html#series-scatter)，[线集](https://www.echartsjs.com/zh/option.html#series-lines)。
17. parallel——[平行坐标系（Parallel Coordinates）](https://en.wikipedia.org/wiki/Parallel_coordinates) 是一种常用的可视化高维数据的图表。
18. parallelAxis——平行坐标系中的坐标轴。
19. singleAxis——单轴
20. timeline——时间线组件，调整图标显示的什么年限的数据
21. graphic——`graphic` 是原生图形元素组件。可以支持的图形元素包括：[image](https://www.echartsjs.com/zh/option.html#graphic.elementsimage), [text](https://www.echartsjs.com/zh/option.html#graphic.elementstext), [circle](https://www.echartsjs.com/zh/option.html#graphic.elementscircle), [sector](https://www.echartsjs.com/zh/option.html#graphic.elementssector), [ring](https://www.echartsjs.com/zh/option.html#graphic.elementsring), [polygon](https://www.echartsjs.com/zh/option.html#graphic.elementspolygon), [polyline](https://www.echartsjs.com/zh/option.html#graphic.elementspolyline), [rect](https://www.echartsjs.com/zh/option.html#graphic.elementsrect), [line](https://www.echartsjs.com/zh/option.html#graphic.elementsline), [bezierCurve](https://www.echartsjs.com/zh/option.html#graphic.elementsbezierCurve), [arc](https://www.echartsjs.com/zh/option.html#graphic.elementsarc), [group](https://www.echartsjs.com/zh/option.html#graphic.elementsgroup),
22. calendar——日历坐标系组件
23. dataset
24. aria
25. series——系列列表。每个系列通过 `type` 决定自己的图表类型，
    - line  折线/面积图
    - bar  柱状图/条形图
    - pie 饼图
    - scatter  散点/气泡图
    - radar 雷达图
    - tree  树图主要用来可视化树形数据结构
    - treemap 它主要用面积的方式，突出展现树的各层级中重要的节点
    - sunburst  旭日图，内圈是外圈的父节点
    - boxplot  箱形图，它能显示出一组数据的最大值、最小值、中位数、下四分位数及上四分位数。
    - candlestick K线图，它能显示最低价，开盘价，收盘价，最高价、
    - heatmap  热力图主要通过颜色去表现数值的大小，必须要配合 [visualMap](https://www.echartsjs.com/zh/option.html#visualMap) 组件使用。
    - map  地图主要用于地理区域数据的可视化，配合 [visualMap](https://www.echartsjs.com/zh/option.html#visualMap) 组件用于展示不同区域的人口分布密度等数据。
    - parallel  平行坐标系的系列。
    - lines  线图，用于带有起点和终点信息的线数据的绘制，主要用于地图上的航线，路线的可视化。
    - graph 关系图，用于展现节点以及节点之间的关系数据。
    - sankey  桑基图，是一种特殊的流程图（可以看作是有向无环图）
    - funnel  漏斗图，可视化分解为阶段的序列数据
    - gauge  仪表盘，
    - pictorialBar  象形柱图，象形柱图是可以设置各种具象图形元素（如图片、[SVG PathData](http://www.w3.org/TR/SVG/paths.html#PathData) 等）的柱状图。
    - themeRiver  主题河流，是一种特殊的流图, 它主要用来表示事件或主题等在一段时间内的变化。
    - custom 自定义系列，可以自定义系列中的图形元素渲染。从而能扩展出不同的图表。
26. color——调色盘颜色列表。
27. backgroundColor
28. textStyle——全局的字体样式。
29. animation
30. blendMode
31. hoverLayerThreshold
32. useUTC