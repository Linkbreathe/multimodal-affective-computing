# Input tables

- Raw XDF and Unity/face/head streams are external, read-only inputs.
- The source workbook supplies participant, Condition, presentation order, and
  questionnaire labels.
- Relative source locations are resolved from the YAML file that declares
  them; production paths are supplied by untracked `local.yaml`.
