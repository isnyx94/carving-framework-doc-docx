"""Inspect NapierOne DOCX-tiny (SHA-256 verified): writer application, data descriptors, text."""
import sys, os, glob, json, zipfile, re, collections
sys.path.insert(0, ".")
import classify.restore.docx_restore as dr
from classify.restore.text_recall import reference_text
D = "data/napierone_docx/extracted"
rows = []
for p in sorted(glob.glob(D + "/*.docx")):
    d = open(p, "rb").read(); r = {"file": os.path.basename(p), "size": len(d)}
    try:
        z = zipfile.ZipFile(p); names = z.namelist()
        r["entries"] = len(names); r["bit3_entries"] = sum(1 for i in z.infolist() if i.flag_bits & 8)
        app = z.read("docProps/app.xml").decode("utf8", "ignore") if "docProps/app.xml" in names else ""
        m = re.search(r"<Application>(.*?)</Application>", app); r["application"] = m.group(1) if m else ""
        m = re.search(r"<AppVersion>(.*?)</AppVersion>", app); r["app_version"] = m.group(1) if m else ""
        core = z.read("docProps/core.xml").decode("utf8", "ignore") if "docProps/core.xml" in names else ""
        r["has_creator_tool_python_docx"] = "python-docx" in core
        r["ok_zip"] = True
    except Exception as e:
        r["ok_zip"] = False; r["error"] = str(e)[:80]
    hs = dr.find_local_headers(d, 512); r["headers_found"] = len(hs)
    r["header_bit3"] = sum(1 for h in hs if h.flags & 8)
    try: r["ref_tokens"] = len(re.findall(r"[A-Za-z0-9']+", reference_text(p)))
    except Exception as e: r["ref_tokens"] = -1
    rows.append(r)
json.dump(rows, open("writing/reanalysis_2026-10-08/napierone_inspect.json", "w"), indent=1)
c = collections.Counter((r.get("application", "?"), r.get("app_version", "")) for r in rows)
print(len(rows), "files"); print(c.most_common())
print("bad zip", sum(not r["ok_zip"] for r in rows))
print("files with any bit-3 entry", sum(r.get("bit3_entries", 0) > 0 for r in rows))
print("files with text", sum(r["ref_tokens"] > 0 for r in rows), "python-docx", sum(r.get("has_creator_tool_python_docx", False) for r in rows))
import statistics as st; print("size median MB", st.median(r["size"] for r in rows) / 1e6, "max", max(r["size"] for r in rows) / 1e6)
