# Remotion 模式

生成 Remotion 视频组合的常见模式和最佳实践。

## 项目设置

### 初始化 Remotion

```bash
npx create-video@latest
```

选择模板：**Blank** 或 **Hello World**

### 项目结构

```
remotion-video/
├── src/
│   ├── compositions/
│   │   ├── session-01/
│   │   │   ├── index.tsx
│   │   │   ├── components/
│   │   │   └── config.ts
│   │   └── session-02/
│   ├── components/        # 共享组件
│   ├── utils/            # 辅助函数
│   └── Root.tsx          # 组合注册
├── public/
│   ├── assets/
│   │   ├── images/
│   │   ├── audio/
│   │   └── fonts/
│   └── bgm/
├── package.json
└── remotion.config.ts
```

## 常见组合

### 标题屏幕

```tsx
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from '@remotion/player';

export const TitleScreen = ({ title, subtitle }: { title: string; subtitle?: string }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 30], [0, 1], { extrapolateRight: 'clamp' });
  const titleScale = spring({ frame, fps, config: { damping: 100 } });

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <div style={{ opacity: titleOpacity, transform: `scale(${titleScale})` }}>
        <h1 style={{ fontSize: 80, color: 'white', textAlign: 'center' }}>
          {title}
        </h1>
        {subtitle && (
          <h2 style={{ fontSize: 40, color: 'rgba(255,255,255,0.8)', marginTop: 20 }}>
            {subtitle}
          </h2>
        )}
      </div>
    </AbsoluteFill>
  );
};
```

### 图片配文字

```tsx
import { AbsoluteFill, Img, interpolate, useCurrentFrame } from '@remotion/player';

export const ImageText = ({
  imageSrc,
  text,
  layout = 'left'
}: {
  imageSrc: string;
  text: string;
  layout?: 'left' | 'right' | 'full';
}) => {
  const frame = useCurrentFrame();
  const textOpacity = interpolate(frame, [15, 30], [0, 1], { extrapolateRight: 'clamp' });

  const imageWidth = layout === 'full' ? '100%' : '60%';
  const textWidth = layout === 'full' ? '0%' : '40%';

  return (
    <AbsoluteFill style={{ flexDirection: 'row' }}>
      {layout !== 'right' && (
        <div style={{ width: imageWidth }}>
          <Img src={imageSrc} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        </div>
      )}
      {layout !== 'full' && (
        <div
          style={{
            width: textWidth,
            padding: 40,
            display: 'flex',
            alignItems: 'center',
            backgroundColor: '#1a1a1a',
          }}
        >
          <p style={{ fontSize: 32, color: 'white', opacity: textOpacity }}>
            {text}
          </p>
        </div>
      )}
      {layout === 'right' && (
        <div style={{ width: imageWidth }}>
          <Img src={imageSrc} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        </div>
      )}
    </AbsoluteFill>
  );
};
```

### 代码展示

```tsx
import { AbsoluteFill, interpolate, useCurrentFrame } from '@remotion/player';

export const CodeDisplay = ({ code, language = 'javascript' }: { code: string; language?: string }) => {
  const frame = useCurrentFrame();
  const lines = code.split('\n');

  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#1e1e1e',
        padding: 40,
        fontFamily: 'monospace',
      }}
    >
      <div style={{ fontSize: 28, color: '#d4d4d4', lineHeight: 1.6 }}>
        {lines.map((line, i) => {
          const lineOpacity = interpolate(
            frame,
            [i * 5, i * 5 + 10],
            [0, 1],
            { extrapolateRight: 'clamp' }
          );
          return (
            <div key={i} style={{ opacity: lineOpacity }}>
              <span style={{ color: '#858585', marginRight: 20 }}>{i + 1}</span>
              {line}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```

### 动画列表

```tsx
import { AbsoluteFill, spring, useCurrentFrame, useVideoConfig } from '@remotion/player';

export const AnimatedList = ({ items }: { items: string[] }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#0f0f0f',
        padding: 60,
        justifyContent: 'center',
      }}
    >
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {items.map((item, i) => {
          const delay = i * 15;
          const scale = spring({
            frame: frame - delay,
            fps,
            config: { damping: 12, stiffness: 200 }
          });

          return (
            <li
              key={i}
              style={{
                fontSize: 48,
                color: 'white',
                marginBottom: 30,
                transform: `scale(${scale})`,
                opacity: frame > delay ? 1 : 0,
              }}
            >
              <span style={{ color: '#667eea', marginRight: 20 }}>●</span>
              {item}
            </li>
          );
        })}
      </ul>
    </AbsoluteFill>
  );
};
```

## 音频集成

### 背景音乐

```tsx
import { Audio, useVideoConfig } from '@remotion/player';

export const WithBGM = ({ src, volume = 0.3 }: { src: string; volume?: number }) => {
  const { durationInFrames } = useVideoConfig();

  return (
    <Audio
      src={src}
      volume={volume}
      startFrom={0}
      endAt={durationInFrames}
    />
  );
};
```

### 配音旁白

```tsx
import { Audio, Sequence, useVideoConfig } from '@remotion/player';

export const NarrationSequence = ({
  audioClips
}: {
  audioClips: Array<{ src: string; startFrame: number; duration: number }>
}) => {
  return (
    <>
      {audioClips.map((clip, i) => (
        <Sequence
          key={i}
          from={clip.startFrame}
          durationInFrames={clip.duration}
        >
          <Audio src={clip.src} volume={1} />
        </Sequence>
      ))}
    </>
  );
};
```

## 转场效果

### 淡入淡出转场

```tsx
import { AbsoluteFill, interpolate, useCurrentFrame } from '@remotion/player';

export const FadeTransition = ({
  children,
  duration = 15
}: {
  children: React.ReactNode;
  duration?: number;
}) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(
    frame,
    [0, duration, duration, duration * 2],
    [0, 1, 1, 0],
    { extrapolateRight: 'clamp' }
  );

  return (
    <AbsoluteFill style={{ opacity }}>
      {children}
    </AbsoluteFill>
  );
};
```

### 滑动转场

```tsx
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from '@remotion/player';

export const SlideTransition = ({
  children,
  direction = 'left'
}: {
  children: React.ReactNode;
  direction?: 'left' | 'right' | 'up' | 'down';
}) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();

  const positions = {
    left: { start: -width, end: 0 },
    right: { start: width, end: 0 },
    up: { start: -height, end: 0 },
    down: { start: height, end: 0 },
  };

  const pos = positions[direction];
  const translateX = direction === 'left' || direction === 'right'
    ? interpolate(frame, [0, 20], [pos.start, pos.end], { extrapolateRight: 'clamp' })
    : 0;
  const translateY = direction === 'up' || direction === 'down'
    ? interpolate(frame, [0, 20], [pos.start, pos.end], { extrapolateRight: 'clamp' })
    : 0;

  return (
    <AbsoluteFill style={{ transform: `translate(${translateX}px, ${translateY}px)` }}>
      {children}
    </AbsoluteFill>
  );
};
```

## 会话组合

将组件组合成完整的会话：

```tsx
// src/compositions/session-01/index.tsx
import { Composition, Sequence } from '@remotion/player';
import { TitleScreen } from '../../components/TitleScreen';
import { ImageText } from '../../components/ImageText';
import { AnimatedList } from '../../components/AnimatedList';
import { WithBGM } from '../../components/WithBGM';
import config from './config.json';

export const Session01 = () => {
  const fps = 30;
  let currentFrame = 0;

  return (
    <AbsoluteFill>
      <WithBGM src="/bgm/track-1.mp3" volume={0.2} />

      {config.scripts.map((script, i) => {
        const startFrame = currentFrame;
        const duration = script.duration * fps;
        currentFrame += duration;

        return (
          <Sequence
            key={i}
            from={startFrame}
            durationInFrames={duration}
          >
            {script.type === 'title' && (
              <TitleScreen title={script.content} />
            )}
            {script.type === 'image-text' && (
              <ImageText
                imageSrc={script.resources[0]}
                text={script.content}
              />
            )}
            {script.type === 'list' && (
              <AnimatedList items={script.content.split('\n')} />
            )}
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

export const registerSession01 = () => (
  <Composition
    id="session-01"
    component={Session01}
    durationInFrames={config.duration * 30}
    fps={30}
    width={1920}
    height={1080}
  />
);
```

## 渲染

### 渲染单个会话

```bash
npx remotion render session-01 out/session-01.mp4
```

### 渲染所有会话

```bash
# 创建渲染脚本
for session in session-01 session-02 session-03; do
  npx remotion render $session out/$session.mp4
done
```

### 合并会话

```bash
# 创建 sessions.txt
file 'out/session-01.mp4'
file 'out/session-02.mp4'
file 'out/session-03.mp4'

# 拼接
ffmpeg -f concat -i sessions.txt -c copy out/final-video.mp4
```

## 最佳实践

1. **保持组件可复用**：通用 props、可配置样式
2. **使用 TypeScript**：为配置和 props 使用强类型
3. **优化图片**：添加到 public/ 前进行压缩
4. **测试时长**：开发过程中频繁预览
5. **音频同步**：仔细计算帧偏移
6. **错误处理**：渲染前验证 config.json
7. **性能优化**：懒加载图片、使用 Sequence 进行内存管理

## 常见问题

### 音视频同步
- 基于 fps 计算帧数（30 fps = 每秒 30 帧）
- 使用 Sequence 组件进行精确计时
- 先用 remotion preview 测试

### 图片加载
- 将图片放在 public/ 文件夹
- 使用绝对路径引用：`/assets/image.png`
- 预加载大图片

### 字体加载
- 将字体添加到 public/fonts/
- 使用 CSS @font-face 或 loadFontsAsync
- 在预览中测试字体渲染

### 内存问题
- 将长视频拆分为多个会话
- 使用 Sequence 组件
- 在渲染之间清理未使用的资源
