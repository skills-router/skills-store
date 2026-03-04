---
name: content-to-video
description: 将文章、论文、GitHub 仓库等各种内容源转换为专业视频。当用户想要将文本内容（URL、文档、代码仓库）转换为视频格式时使用。涵盖研究、脚本编写、场景规划和基于 Remotion 的视频生成。触发词包括"将这篇文章转换为视频"、"从这篇论文创建视频"或"制作一个关于这个 GitHub 项目的视频"。
---

# 内容转视频

将各种内容源（文章、论文、GitHub 仓库等）转换为专业视频，包含结构化研究、脚本编写和基于 Remotion 的生成工作流程。

## 工作流程

### 1. 收集需求

首先需要查看 remotion-best-practices 技能，这是视频生成的核心技能

询问用户：
- **内容来源**：URL、文件路径或主题描述
- **目标受众**：开发者、普通大众、学生等
- **视频风格**：教育类、推广类、纪录片类、教程类
- **视觉风格**：Apple 发布会风格（极简科技）、科技创业路演风格（TED / 创业发布）、互联网产品宣传风格（Figma / Notion 风格）、纪录片风格（Netflix 风格）、快节奏短视频风格（TikTok / Reels 风格）
- **格式**：横屏（16:9）、竖屏（9:16）、方形（1:1）
- **语言**：配音和字幕语言，是否需要多语言版本的视频

创建项目目录：`<topic-slug>/`
将所有用户需求写入 `<topic-slug>/video-style.md`

### 2. 深度研究

创建 `<topic-slug>/deep-research/` 目录存储：
- 源内容（下载的文章、抓取的网页）
- 从源内容提取的资源：图片、视频（下载、AI 生成、截图）
- 额外资源的生成和创建
- BGM 选项（搜索和下载，详见 free-resource 技能）
- 关键洞察和数据点
- 最后为该目录生成 Readme.md 以备后用

**按来源类型处理：**
- **URL**：获取完整内容、提取要点、下载相关图片
- **论文**：解析 PDF、提取摘要/方法/结果、生成可视化
- **GitHub**：分析仓库结构、README、代码亮点、生成图表

详见 [references/research-workflow_CN.md](references/research-workflow_CN.md)。

### 3. 规划视频结构

创建 `<topic-slug>/video-sessions/` 目录，包含子目录：
- `session-01/`、`session-02/` 等

每个会话包含：
- `config.json` - 会话配置（见下方 schema）
- `assets/` - 会话特定的图片、视频、音频

**会话 config.json schema：**
```json
{
  "title": "会话标题",
  "scripts": [
    {
      "type": "composition_type",
      "content": "文本或内容描述",
      "transition": "fade",
      "resources": ["path/to/image.png", "path/to/video.mp4"],
      "narration": "配音文本",
      "prompts": "AI 视觉布局提示词、展示风格、要生成的元素和文本、所用资源的详细提示词"
    }
  ]
}
```

完整 schema 见 [references/session-config-schema_CN.md](references/session-config-schema_CN.md)。

### 4. 用户审核

展示完整的视频结构：
- 总时长和会话分解
- 会话标题和要点
- 示例配音文本
- 预期视觉风格

在继续之前获取用户批准。允许迭代：
- 会话顺序和内容
- 配音语调和长度
- 视觉风格偏好

### 5. 使用 Remotion 生成

创建 `<topic-slug>/remotion-video/` 目录：
- 初始化 Remotion 项目
- 创建会话组合：`src/compositions/session-01/` 等
- 生成音频配音（TTS 或用户提供），使用 qwen3-audio 技能，生成的音频返回时长，注意在每个会话切换前后预留空白过渡时间，必须等待音频播放完毕后再播放下一个会话或动画
- 添加资源到 `public/assets/`
- 配置转场和效果

**会话结构：**
```
session-01/
├── index.tsx          - 主组合
├── components/        - 会话特定组件
├── assets/           - 音频、图片
└── config.ts         - 会话配置
```

常见模式见 [references/remotion-patterns_CN.md](references/remotion-patterns_CN.md)。



## 目录结构

最终项目结构：
```
<topic-slug>/
├── deep-research/
│   ├── content/
│   ├── images/
│   ├── bgm/
│   └── notes.md
├── video-sessions/
│   ├── session-01/
│   │   ├── config.json
│   │   └── assets/
│   ├── session-02/
│   └── ...
├── remotion-video/
│   ├── src/
│   ├── public/
│   └── package.json
└── output/
    └── final-video.mp4
```

## 关键注意事项

- **Token 效率**：研究阶段可能消耗大量 token。使用增量保存和摘要。
- **并行工作**：可以在研究的同时准备 Remotion 模板。
- **迭代**：预计在生成视频前需要 2-3 轮用户对脚本的反馈。
- **质量与速度**：提前询问用户偏好 - 快速原型还是精良成品。
