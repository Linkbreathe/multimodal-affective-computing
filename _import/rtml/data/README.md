# Data entry contract

This directory never contains copied raw XDF, CSV, video, or questionnaire
files.  Their machine-specific roots belong in the untracked
`configs/local.yaml`.

`contracts/` describes the stable inputs and outputs consumed by the pipeline.
The supervised unit remains one participant--Condition row (15 x 9 = 135),
not an individual inherited-label 10-second window.
