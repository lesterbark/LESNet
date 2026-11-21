"""Assemble the paper.

    run.bat -u build_paper.py     ->  PAPER.md (source of truth) and PAPER.html (self-contained, images inlined)

`text/NN_*.md` are concatenated in order.  `{{table:NAME}}` is replaced by the table `NAME` that
aggregate.py wrote into results/tables.md, and `{{fig:NAME|caption}}` by an image from figures/ with a
caption, so every number in the paper comes from a results file, not from this folder's prose.
"""
import base64
import glob
import os
import re

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))


def tables():
    p = os.path.join(HERE, "results", "tables.md")
    out = {}
    if os.path.exists(p):
        txt = open(p, encoding="utf-8").read()
        for m in re.finditer(r"<!-- table:(\S+) -->\n(.*?)<!-- /table -->", txt, re.S):
            out[m.group(1)] = m.group(2).strip()
    mdir = os.path.join(HERE, "tables_manual")
    if os.path.isdir(mdir):
        for f in os.listdir(mdir):
            out[os.path.splitext(f)[0]] = open(os.path.join(mdir, f), encoding="utf-8").read().strip()
    return out


def build():
    T = tables()
    parts = [open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(HERE, "text", "*.md")))]
    md = "\n\n".join(p.strip() for p in parts) + "\n"
    # numbering: tables and figures are numbered in order of appearance; {{ref:name}} cites them
    num, counters = {}, {"Table": 0, "Figure": 0}
    for m in re.finditer(r"\{\{(table|fig):([^|}]+?)(?:\|.*?)?\}\}", md, flags=re.S):
        kind = "Table" if m.group(1) == "table" else "Figure"
        if m.group(2) not in num:
            counters[kind] += 1
            num[m.group(2)] = f"{kind} {counters[kind]}"

    def table(m):
        name, cap = m.group(1), (m.group(2) or "").strip()
        body = T.get(name, f"*(table {name} not generated)*")
        head = f"**{num[name]}.** {cap}" if cap else f"**{num[name]}.**"
        return head + "\n\n" + body

    md = re.sub(r"\{\{table:([^|}]+?)(?:\|(.*?))?\}\}", table, md, flags=re.S)
    md = re.sub(r"\{\{fig:([^|}]+?)\|(.*?)\}\}",
                lambda m: f"![{num[m.group(1)]}](figures/{m.group(1)})" + "\n\n" + f"**{num[m.group(1)]}.** {m.group(2)}",
                md, flags=re.S)
    md = re.sub(r"\{\{ref:([^}]+?)\}\}", lambda m: num.get(m.group(1), "??"), md)
    open(os.path.join(HERE, "PAPER.md"), "w", encoding="utf-8").write(md)

    html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])

    def inline(m):
        path = os.path.join(HERE, m.group(1))
        if not os.path.exists(path):
            return m.group(0)
        b = base64.b64encode(open(path, "rb").read()).decode()
        return f'src="data:image/png;base64,{b}"'

    html_body = re.sub(r'src="(figures/[^"]+)"', inline, html_body)
    css = """
    body{font:16px/1.55 Georgia,'Times New Roman',serif;max-width:860px;margin:2.5rem auto;padding:0 1.2rem;color:#1d2330}
    h1{font-size:1.7rem;line-height:1.25} h2{margin-top:2.2rem;border-bottom:1px solid #d5d9e2;padding-bottom:.2rem}
    h3{margin-top:1.6rem} table{border-collapse:collapse;margin:1rem 0;font:13.5px/1.35 system-ui,sans-serif}
    th,td{border:1px solid #d5d9e2;padding:.3rem .55rem;text-align:left} th{background:#f2f4f8}
    code,pre{font:13px Consolas,monospace;background:#f4f5f8} pre{padding:.7rem;overflow:auto} code{padding:0 .2rem}
    img{max-width:100%} em{color:#4a5368} blockquote{border-left:3px solid #d1495b;margin:1rem 0;padding:.1rem 1rem;color:#333}
    @media print{
      body{max-width:none;margin:0;padding:0;font-size:11pt;line-height:1.45}
      h1,h2,h3{break-after:avoid;page-break-after:avoid}
      table,pre,blockquote,img{break-inside:avoid;page-break-inside:avoid}
      table{font-size:9pt}
    }
    """
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>LESNet</title><style>{css}</style></head><body>{html_body}</body></html>"
    open(os.path.join(HERE, "PAPER.html"), "w", encoding="utf-8").write(html)
    print("wrote PAPER.md and PAPER.html:", len(md.split()), "words")


if __name__ == "__main__":
    build()
