window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']], // 支持 `$...$` 和 `\(...\)` 作为行内公式
      displayMath: [['$$', '$$'], ['\\[', '\\]']] // 支持 `$$...$$` 作为块级公式
    },
    svg: {
      fontCache: 'global'
    }
  };

  // 监听文档加载事件，确保公式正确渲染
  document.addEventListener("DOMContentLoaded", function() {
      MathJax.typesetPromise();

      // 监听所有折叠块，确保 `<details>` 或 `<div>` 里的公式被解析
      document.querySelectorAll("details, .theorem-box").forEach(el => {
          el.addEventListener("toggle", () => {
              MathJax.typesetPromise([el]);
          });
      });
  });
