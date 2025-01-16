# Generative AI: From Start to Surrender

## Explore the Documentation Online :book:

Discover the full documentation for "Generative AI: From Start to Surrender" online. Dive into detailed guides and resources to enhance your understanding of generative AI technologies.

[![Read Online](https://img.shields.io/badge/Read-Online-blue?style=for-the-badge)](https://mad-sg.github.io/generative-ai-start-to-surrender/)

## Contributor :busts_in_silhouette:

We are grateful for the contributions from our community. Here are some of our key contributors:

- [![GitHub](https://img.shields.io/badge/GitHub-Qian%20Lilong-lightgrey?logo=github&style=social)](https://github.com/tsiendragon)

Your contributions help make this project better. Thank you for your support!

## Contribution
### vscode 设置保存图片位置. vscode 可以支持直接复制图片，会把粘贴板上的图片复制到markdown 文件中，并且保存文件到指定目录中。

在settings.json中添加

```json
{
    "markdown.copyFiles.destination": {
        "**/*": "${documentWorkspaceFolder}/docs/images/"
    },
    ...
}
```

### 使用Markdownlint

Markdownlint 是一个对 Markdown 进行规范检查的扩展，除了给出 lint 提示外，也支持自动修复部分问题（比如空行、列表缩进等）。
打开 VS Code 设置后，可根据需要启用自动修复选项：

```json
{
   ...
   "editor.codeActionsOnSave": {
      "source.fixAll.markdownlint": true
   }
}
```

## BUGS

### MARKDOWN

   1. bmatrix cannot be rendered
   2. Use extension "MathJax 3 Plugin for Github" to render equation![alt text](docs/images/image-12.png)
   3. cannot render \mathbf

   4. $$ $$ 需要换行。在 Markdown 中，单个换行符（按下 Enter）通常不会被视为换行，而是被解析为一个空格，除非渲染器明确支持换行。


   行间公式保持下面的格式

   ```markdown
   text...

   $$...$$



   ```

## License

This repository contains both the book content and source code. Each is licensed separately:

1. **Book Content**:

   - "Generative AI: From Start to Surrender – A Practical Guide to Mastering and Struggling with AI Models"  2025.
   - Licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
2. **Code**:

   - Licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

Please ensure you follow the respective license terms when using this material.
