---
term: "MLOps"
summary: "The set of practices for taking a machine learning model from a
notebook to a reliably running production system — training pipelines,
serving infrastructure, monitoring, and retraining, treated with the same
rigor as regular software engineering (CI/CD, testing, observability)."
tag: "Machine Learning"
---

MLOps borrows its name and much of its philosophy from DevOps: the goal is
to make model training, deployment, and monitoring repeatable and
automatable rather than a one-off notebook exercise.

In practice this usually covers: versioned training pipelines, automated
evaluation gates before a model is promoted, a serving layer with
monitoring for data/prediction drift, and a retraining loop that kicks in
when performance degrades. It's distinct from **MLaaS** (ML-as-a-Service),
where a team consumes a hosted foundation model via an API instead of
running its own training/serving stack.
