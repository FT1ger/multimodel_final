# Report

Files:

- `report.tex`: full report source.
- `references.bib`: BibTeX entries.
- `report.pdf`: compiled report (12 pages).
- `figures/`: aggregate report figures (PDF and PNG).

TinyTeX is installed at `~/Library/TinyTeX`. Compile from this directory:

```bash
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
pdflatex report
bibtex report
pdflatex report
pdflatex report
```

This report contains no raw tweets or images; all figures are aggregate
statistics or schematic diagrams.
