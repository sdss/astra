---
hide-toc: true
---

# Astra

The aim of this guide is to provide a comprehensive introduction on how Astra works.
The expected audience has experience with Python, and falls into one or more of the following categories:
- intends to add an existing analysis pipeline as a component in Astra;
- wants to create or update a directed acyclic graph (DAG) for using Astra and Airflow; or
- intends to contribute to the general Astra source code.

This guide starts with the fundamentals of Astra. 
A [task is introduced](tasks) as a reproducible unit of work, with different [parameters](parameters) that might change the expected output. We'll describe how tasks with common overheads are [bundled together](bundles) for efficiency, and how all tasks can be reproduced because the input [data products](data-products) and [summary outputs](outputs) are [saved to a database](database), mostly automatically. This allows us to create summary tables of results, compare outputs from different analysis methods, or code versions.

This general framework is comprehensive enough that the reader should be able to write their own executable task, and use it in Astra. And you could use this for any data analysis purpose: there's no requirement that the tasks need to have anything to do with astronomy. They could use that task to analyse data that is stored locally, with the inputs and summary outputs stored in a local database. You'd just have to write some scripts to create and execute the tasks with some frequency, which you could do through a `cron` job (or similar). 

An individual task represents the smallest unit of work in Astra. Usually you want to use the outputs from one task as inputs to another task, or you want to execute many tasks in a complex graph. In that case, sometimes a simple `cron` job is not enough, and you need a workflow to schedule and execute tasks following some logic. There are many tools that can do this orchestration (e.g., [luigi](#), [airflow](#), [snakemake](#), and others). You can use any of these with Astra. After trying a few others, we [adopted Airflow in SDSS-V](airflow-index). 


### Contents

```{toctree}
tasks
parameters
data-products
outputs
bundles
database
```
