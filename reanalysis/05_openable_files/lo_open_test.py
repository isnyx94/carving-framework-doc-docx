"""Open each rebuilt .docx in LibreOffice (headless, via UNO) and record whether it loads and how much
text it shows. Resumable: results are appended per file. Usage: python lo_open_test.py <dir> <out.jsonl> <seconds>"""
import uno, sys, os, json, time, subprocess
from com.sun.star.beans import PropertyValue
d, out, budget = sys.argv[1], sys.argv[2], float(sys.argv[3]); t0 = time.time()
done = set()
if os.path.exists(out):
    for l in open(out):
        try: done.add(json.loads(l)["file"])
        except Exception: pass
def pv(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v; return p
def start():
    subprocess.run(["pkill", "-x", "soffice.bin"])
    proc = subprocess.Popen(["soffice", "--headless", "--invisible", "--norestore", "--accept=socket,host=localhost,port=2003;urp;"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    local = uno.getComponentContext(); res = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(60):
        try:
            ctx = res.resolve("uno:socket,host=localhost,port=2003;urp;StarOffice.ComponentContext"); break
        except Exception: time.sleep(0.5)
    return proc, ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
proc, desk = start()
with open(out, "a") as fh:
    for f in sorted(os.listdir(d)):
        if f in done: continue
        if time.time() - t0 > budget: break
        rec = {"file": f, "loads": False, "chars": 0}
        try:
            doc = desk.loadComponentFromURL(uno.systemPathToFileUrl(os.path.join(d, f)), "_blank", 0, (pv("Hidden", True), pv("ReadOnly", True)))
            if doc is not None:
                rec["loads"] = True
                try: rec["chars"] = len(doc.getText().getString())
                except Exception: pass
                doc.close(True)
        except Exception as e:
            rec["error"] = type(e).__name__
            try: proc.kill()
            except Exception: pass
            proc, desk = start()
        fh.write(json.dumps(rec) + "\n"); fh.flush()
print("done", sum(1 for _ in open(out)))
proc.kill()
