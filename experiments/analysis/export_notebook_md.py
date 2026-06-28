"""Exporta un .ipynb a Markdown 100% fiel (code + outputs de texto), SIN modificar el original.

- No altera el notebook fuente (solo lectura).
- Copia literalmente el source de cada celda y sus outputs de texto
  (stdout/stderr, execute_result/display_data text/plain, errores/tracebacks).
- Las salidas de imagen se sustituyen por un marcador (no se pierde info numérica).
- Las tablas HTML de pandas se omiten en favor del text/plain (que pandas siempre
  incluye y es idéntico a lo que se ve en consola); si no hubiera text/plain, se avisa.
"""
import json
import sys
import re
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("MAVIS_master_analysis.ipynb")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC.with_suffix(".export.md")

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def text_of(field):
    return field if isinstance(field, str) else "".join(field)


def render_output(out):
    t = out.get("output_type")
    lines = []
    if t == "stream":
        lines.append(f"[stream:{out.get('name','stdout')}]")
        lines.append(text_of(out.get("text", "")).rstrip("\n"))
    elif t in ("execute_result", "display_data"):
        data = out.get("data", {})
        if "text/plain" in data:
            tag = "execute_result" if t == "execute_result" else "display_data"
            lines.append(f"[{tag} text/plain]")
            lines.append(text_of(data["text/plain"]).rstrip("\n"))
        kinds = [k for k in data if k.startswith("image/")]
        if kinds:
            lines.append(f"[image output omitted: {', '.join(kinds)}]")
        if "text/plain" not in data and not kinds:
            lines.append(f"[output with keys: {', '.join(data.keys())}]")
    elif t == "error":
        lines.append(f"[error: {out.get('ename','')}: {out.get('evalue','')}]")
        tb = "\n".join(out.get("traceback", []))
        lines.append(ANSI.sub("", tb).rstrip("\n"))
    else:
        lines.append(f"[unknown output_type: {t}]")
    return "\n".join(lines)


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]
    parts = [
        f"# Export fiel de `{SRC.name}`",
        "",
        f"Fuente: `{SRC}`  ·  {len(cells)} celdas  ·  nbformat {nb.get('nbformat')}.{nb.get('nbformat_minor')}",
        "",
        "> Generado por `export_notebook_md.py`. Copia literal de code + outputs de texto. "
        "Las imágenes se marcan como omitidas; ningún número del output ha sido alterado.",
        "",
        "---",
        "",
    ]
    for i, c in enumerate(cells):
        ct = c["cell_type"]
        src = text_of(c.get("source", "")).rstrip("\n")
        ec = c.get("execution_count")
        header = f"## Celda {i} · {ct}"
        if ct == "code" and ec is not None:
            header += f" · exec[{ec}]"
        parts.append(header)
        parts.append("")
        if ct == "markdown":
            parts.append(src if src else "_(markdown vacío)_")
        elif ct == "code":
            parts.append("```python")
            parts.append(src)
            parts.append("```")
            outs = c.get("outputs", [])
            if outs:
                parts.append("")
                parts.append("**Output:**")
                parts.append("")
                parts.append("```text")
                parts.append("\n\n".join(render_output(o) for o in outs))
                parts.append("```")
        else:
            parts.append(f"_(tipo de celda no soportado: {ct})_")
        parts.append("")
        parts.append("---")
        parts.append("")

    DST.write_text("\n".join(parts), encoding="utf-8")
    print(f"OK -> {DST}")
    print(f"   bytes: {DST.stat().st_size}")


if __name__ == "__main__":
    main()
