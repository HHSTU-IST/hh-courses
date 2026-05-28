# 前端基础

<link rel="stylesheet" type="text/css" href="styles/zero.css">

## 交互元素

### 折叠

```html
<details>
  <summary>标题</summary>
  被折叠内容
</details>
```

<details>
  <summary>《静夜思》作者：李白</summary>
  床前明月光，疑是地上霜。举头望明月，低头思故乡。
</details>

### 图片尺寸

```html
<img src="https://pic2.zhimg.com/80/v2-48bbd284deacef0b5896427e660b2a51_1440w.png" width="400" height="200">
```

<img src="https://pic2.zhimg.com/80/v2-48bbd284deacef0b5896427e660b2a51_1440w.png" width="400" height="200" alt="">

### 音视频

```html
<video controls width="100%">
  <source src="本地路径|网址" type="video/mp4">
</video>
```

<video controls width="100%">
  <source src="img/sorting.mp4" type="video/mp4">
</video>

音频同理

```html
<audio controls>
  <source src="本地路径|网址" type="audio/mpeg">
</audio>
```

## AI 辅助学习

### 生成 HTML 控件

```markdown
<!-- 第 1 次 -->
在 HTML 中，生成方程组求解器，要求
- 能随机生成方程组
- 能调整方程组数量
<!-- 第 2 次 -->
方程组数量无法调整，请修正对应代码
```

### 生成 CSS 样式

```markdown
<!-- 第 1 次 -->
生成 Markdown CSS 样式，要求
- 兼容 Pandoc
- 暗色主题，赛博风格
<!-- 第 2 次 -->
请降低对比度
```

### 工程化

- 外部引入 CSS 样式

```html
<link rel="stylesheet" type="text/css" href="样式路径">
```

- 外部引入 JS 脚本

```html
<script src="脚本路径"></script>
```
