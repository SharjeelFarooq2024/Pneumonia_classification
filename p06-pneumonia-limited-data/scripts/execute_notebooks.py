"""Execute a jupyter notebook and embed outputs directly into the .ipynb JSON."""
import os
import sys
import json
import io
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import Image as IPImage

def execute_notebook(nb_path: Path):
    print(f"--- Executing {nb_path.name} ---")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Reset IPython InteractiveShell
    shell = InteractiveShell.instance()
    shell.reset()
    
    # Custom display capture
    displayed_items = []
    
    def custom_display(*objs, **kwargs):
        for obj in objs:
            if isinstance(obj, IPImage):
                img_path = obj.filename
                if img_path and Path(img_path).exists():
                    with open(img_path, "rb") as img_f:
                        b64 = base64.b64encode(img_f.read()).decode("utf-8")
                    displayed_items.append({
                        "output_type": "display_data",
                        "data": {
                            "image/png": b64,
                            "text/plain": f"<IPython.core.display.Image object>"
                        },
                        "metadata": {}
                    })
            elif hasattr(obj, "_repr_html_"):
                displayed_items.append({
                    "output_type": "display_data",
                    "data": {
                        "text/html": obj._repr_html_(),
                        "text/plain": repr(obj)
                    },
                    "metadata": {}
                })
            else:
                displayed_items.append({
                    "output_type": "display_data",
                    "data": {
                        "text/plain": repr(obj)
                    },
                    "metadata": {}
                })

    # Monkeypatch display in builtins and IPython
    import builtins
    builtins.display = custom_display
    import IPython.display
    IPython.display.display = custom_display

    exec_count = 1
    for cell_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        
        source = "".join(cell["source"])
        if not source.strip():
            continue

        displayed_items.clear()
        
        with capture_output() as cap:
            res = shell.run_cell(source)
            
        outputs = []
        
        # Check standard output
        stdout_text = cap.stdout
        if stdout_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.splitlines()]
            })
            
        # Check errors
        if res.error_in_exec:
            ename = type(res.error_in_exec).__name__
            evalue = str(res.error_in_exec)
            outputs.append({
                "ename": ename,
                "evalue": evalue,
                "output_type": "error",
                "traceback": [f"{ename}: {evalue}"]
            })
            print(f"  [ERROR] Cell {cell_idx}: {ename}: {evalue}")

        # Check explicit display items
        for item in displayed_items:
            outputs.append(item)

        # Check matplotlib figures
        fig_nums = plt.get_fignums()
        if fig_nums:
            for num in fig_nums:
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                outputs.append({
                    "data": {
                        "image/png": b64,
                        "text/plain": f"<Figure size {fig.get_size_inches()[0]*100:.0f}x{fig.get_size_inches()[1]*100:.0f} with {len(fig.axes)} Axes>"
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            plt.close("all")

        # Check cell return value
        if res.result is not None and not res.error_in_exec:
            val = res.result
            data = {}
            if hasattr(val, "_repr_html_"):
                data["text/html"] = val._repr_html_()
            data["text/plain"] = repr(val)
            outputs.append({
                "data": data,
                "execution_count": exec_count,
                "metadata": {},
                "output_type": "execute_result"
            })

        cell["execution_count"] = exec_count
        cell["outputs"] = outputs
        exec_count += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"  Successfully executed and saved {nb_path.name} with {exec_count-1} code cells.")

if __name__ == "__main__":
    for nb_file in sorted((ROOT / "notebooks").glob("*.ipynb")):
        execute_notebook(nb_file)
