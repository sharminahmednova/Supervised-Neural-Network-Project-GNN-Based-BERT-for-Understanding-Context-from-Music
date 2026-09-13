# Final report

`main.tex` is a stand-alone skeleton (compiles with pdflatex) wired to the
generated tables and figures. To submit on an official template, copy the
section bodies across -- the `\input` paths and figure names do not change.

## Regenerate tables and figures

    python src/evaluate.py              # -> results/plots/*.png
    python tools/make_report_tables.py  # -> report/tables/*.tex

Every number in the report comes from `results/metrics.json`, so the tables
cannot drift from the run that produced them. Do not hand-edit `tables/*.tex`.

## Overleaf templates

- NeurIPS 2024: https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh
- IEEE Conference: https://www.overleaf.com/latex/templates/ieee-conference-template
- ICML 2025: https://www.overleaf.com/latex/templates/icml2025-template

## Before submitting

- [ ] 6-10 pages
- [ ] State plainly which corpus produced the numbers (see README section 2.1 --
      if the run used the synthetic corpus, say so and do not present it as
      evidence about FMA or MagnaTagATune)
- [ ] At least two baselines, described with a fair setup
- [ ] Ablation table and t-SNE included
- [ ] 3 case studies with graph paths and attended caption tokens
- [ ] Limitations section is specific, not boilerplate

## Before submitting: resolve every \FILL

`main.tex` is fully written except for a handful of specific numbers that can
only be stated once the GPU run finishes. They are marked with a red
`\FILL{...}` so they are impossible to miss in the compiled PDF:

```bash
grep -n 'FILL' main.tex
```

Each one names exactly what to state and which generated table it comes from.
Delete the `\FILL{...}` and write the sentence. If a `\FILL` is still red in
the PDF, the report is not finished.

## Generation order

```bash
python src/evaluate.py              --set paths.results=results_mtat --set paths.plots=results_mtat/plots
python tools/make_report_tables.py  --results results_mtat
python tools/aggregate_human_eval.py --results results_mtat     # after raters return
```

`tables/*.tex` are generated — never hand-edit them, the next run overwrites
your changes.
