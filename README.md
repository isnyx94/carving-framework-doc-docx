# Carving framework for DOC and DOCX data restoration

Code for the PhD thesis *Developing a Carving Framework for Data Restoration of Composite File-Structure Document Types (DOC and DOCX)* by Nor Ika Shahirah Binti Ramli (Faculty of Computing, Universiti Malaysia Pahang Al-Sultan Abdullah). Supervisors: Syifak Izhar Hisham and Ferda Ernawan.

The framework recovers document text from fragmented disk images by using the internal structure of the two Word containers:

- **DOCX (Office Open XML, a ZIP package):** each part is inflated from its own local header. When a part ends before its declared size, a decoder-guided search looks for the next block.
- **DOC (Compound File Binary Format):** the allocation table and directory are parsed and only the `WordDocument` stream is read.

## Start here

**`CODE_GUIDE.md`** lists every folder and script, with what it does, how important it is, and which thesis section, table or result file it produces.

```
classify/      core framework (restorers, fragmenter, text metric, adapters, classifiers)
reanalysis/    one folder per study (main DOCX studies, FAT32 volumes, production carvers, DOC, ...)
```

## Install and run

```
pip install -r requirements.txt -r requirements-followup.txt
python reanalysis/01_docx_main_studies/rescore_run.py napier shuffled severe 0 B0,B2 out.jsonl
```

Run all scripts from the repository root. See `CODE_GUIDE.md`, section 2, for where to put the data.

## Data (not included)

- **FFT-75** (Mittal, Korus & Memon, 2021): https://doi.org/10.21227/kfxw-8084
- **govdocs1** thread 000 (Garfinkel et al., 2009): https://digitalcorpora.org/corpora/files
- **NapierOne** DOCX subset (Davies, Macfarlane & Buchanan, 2022): put the files in `data/napierone_docx/extracted/`.
- The python-docx corpus of the main study came from previous research and contains third-party personal data, so it is not released.

PhotoRec and Foremost are used as baselines; they are not redistributed.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). Please cite the thesis using `CITATION.cff`.
