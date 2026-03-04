# 会话配置 Schema

视频会话配置文件的完整 schema。

## 概述

每个会话（`session-01/config.json` 等）遵循以下 schema：

```typescript
interface SessionConfig {
  title: string;           // 会话标题
  scripts: ScriptItem[];   // 脚本项数组
  metadata?: {
    keywords?: string[];   // 关键词
    mood?: string;         // 情绪
    pace?: 'slow' | 'medium' | 'fast';  // 节奏
  };
}

interface ScriptItem {
  type: CompositionType;           // 组合类型
  content: string;                 // 内容
  transition?: TransitionType;     // 转场效果
  resources?: string[];            // 资源路径
  narration?: string;              // 配音文本
  prompts?: string;                // AI 视频内容生成提示词
  timing?: {
    start?: number;                // 开始时间
    end?: number;                  // 结束时间
  };
}

type CompositionType =
  | 'title'           // 标题屏幕
  | 'text'            // 背景上的文字
  | 'image'           // 单张图片
  | 'image-text'      // 图片配文字叠加
  | 'video'           // 视频片段
  | 'code'            // 代码片段
  | 'diagram'         // 图表/流程图
  | 'chart'           // 图表/图形
  | 'split'           // 分屏
  | 'list'            // 项目符号列表
  | 'quote'           // 引用高亮
  | 'transition';     // 转场幻灯片

type TransitionType =
  | 'fade'            // 淡入淡出
  | 'slide-left'      // 向左滑动
  | 'slide-right'     // 向右滑动
  | 'slide-up'        // 向上滑动
  | 'slide-down'      // 向下滑动
  | 'zoom-in'         // 放大
  | 'zoom-out'        // 缩小
  | 'wipe'            // 擦除
  | 'none';           // 无转场
```

## 配置示例

### 标题屏幕

```json
{
  "title": "引言",
  "scripts": [
    {
      "type": "title",
      "content": "理解机器学习",
      "transition": "fade",
      "prompts": "深色渐变背景配微妙粒子动画，居中标题配下方副标题，现代感，科技感，专业"
    }
  ]
}
```

### 图片配音

```json
{
  "title": "核心概念",
  "scripts": [
    {
      "type": "image-text",
      "content": "神经网络的灵感来源于人脑",
      "transition": "slide-left",
      "resources": ["assets/neural-network.png"],
      "narration": "神经网络是受人脑中生物神经网络启发而设计的计算系统。",
      "prompts": "神经网络图配动画连接线，图片占左 60%，文字占右 40%"
    }
  ]
}
```

### 代码示例

```json
{
  "title": "实现",
  "scripts": [
    {
      "type": "code",
      "content": "const model = tf.sequential();\nmodel.add(tf.layers.dense({units: 1, inputShape: [1]}));",
      "transition": "fade",
      "narration": "这是如何使用 TensorFlow.js 创建简单神经网络",
      "prompts": "代码编辑器配语法高亮，全屏代码配行号，深色主题，大字体"
    }
  ]
}
```

### 多项会话

```json
{
  "title": "机器学习的优势",
  "scripts": [
    {
      "type": "list",
      "content": "1. 自动化\n2. 预测\n3. 个性化\n4. 规模化",
      "transition": "fade",
      "narration": "机器学习为企业提供四个关键优势。",
      "prompts": "动画列表项逐个出现，居中列表配图标"
    },
    {
      "type": "image-text",
      "content": "自动化减少 80% 的手动工作",
      "duration": 15,
      "transition": "slide-up",
      "resources": ["assets/automation-chart.png"],
      "narration": "研究表明自动化可以减少高达 80% 的手动任务。",
      "prompts": "图表高亮关键数据"
    }
  ],
  "metadata": {
    "keywords": ["自动化", "效率", "规模"],
    "mood": "professional",
    "pace": "medium"
  }
}
```

## 字段指南

### title
- 保持简短（2-5 个词）
- 描述性强，便于导航
- 用于 Remotion 组合命名

### duration
- 所有脚本项时长之和
- 典型会话：20-60 秒
- 完整视频：3-10 分钟

### scripts[].type
根据内容选择：
- **title**：开场/结束屏幕
- **text**：仅配音片段
- **image**：视觉演示
- **image-text**：最常见 - 视觉 + 解释
- **code**：技术教程
- **diagram**：流程解释
- **chart**：数据/统计
- **list**：要点列举
- **quote**：专家引用、推荐

### scripts[].transition
- **fade**：默认，适用于所有场景
- **slide-***：方向性流动
- **zoom-***：强调变化
- **none**：即时切换（谨慎使用）

### scripts[].narration
- 目标每分钟 150 字
- 短句更清晰
- 自然语言，避免术语
- 语调匹配受众

### scripts[].prompts
Remotion 实现指导：
- **visual**：展示什么
- **layout**：如何排列
- **style**：美学方向

## 验证

生成视频前，验证：
1. 总时长与脚本项之和匹配
2. 所有资源路径存在
3. 配音长度适合时长（150 字/分钟）
4. 转场效果兼容
5. 无时间重叠

## 最佳实践

1. **强力开场**：第一个会话抓住注意力
2. **节奏变化**：混合快慢片段
3. **视觉多样性**：不要过于频繁重复相同类型
4. **配音流畅**：大声朗读检查自然度
5. **时间缓冲**：在主要转场之间添加 0.5 秒
