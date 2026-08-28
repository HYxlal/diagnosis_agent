#!/usr/bin/env python3
"""离线渲染 Markdown 为 HTML，浏览器打开"""
import sys, os, webbrowser, markdown

md_path = sys.argv[1] if len(sys.argv) > 1 else "docs/答辩文档.md"
html_path = md_path.rsplit(".", 1)[0] + ".html"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# GitHub 风格渲染，支持表格、代码高亮、fenced code
extensions = ["tables", "fenced_code", "toc", "nl2br", "sane_lists"]
html_body = markdown.markdown(text, extensions=extensions)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{os.path.basename(md_path)}</title>
<style>
:root {{ --fg:#24292e; --bg:#fff; --border:#e1e4e8; --code-bg:#f6f8fa; --link:#0366d6; }}
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  color: var(--fg); background: var(--bg);
  max-width: 900px; margin: 40px auto; padding: 0 24px;
  line-height: 1.7; font-size: 16px;
}}
h1,h2,h3,h4 {{ border-bottom: 1px solid var(--border); padding-bottom: .3em; margin-top: 1.8em; }}
h1 {{ font-size: 2em; }}
h2 {{ font-size: 1.5em; }}
h3 {{ font-size: 1.25em; }}
code {{
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  background: var(--code-bg); padding: .2em .4em; border-radius: 3px; font-size: 90%;
}}
pre {{ background: var(--code-bg); padding: 16px; border-radius: 6px; overflow-x: auto; }}
pre code {{ background: none; padding: 0; }}
blockquote {{ border-left: 4px solid var(--border); padding: 0 1em; color: #6a737d; margin: 1em 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid var(--border); padding: 6px 13px; text-align: left; }}
th {{ background: var(--code-bg); font-weight: 600; }}
tr:nth-child(even) {{ background: var(--code-bg); }}
a {{ color: var(--link); text-decoration: none; }}
hr {{ border: none; border-top: 2px solid var(--border); margin: 2em 0; }}
img {{ max-width: 100%; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"已生成: {html_path}")
try:
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
except Exception as e:
    print(f"浏览器自动打开失败（{e}），请手动打开: {os.path.abspath(html_path)}")
