# metasignal — Library Structure

```mermaid
graph TD
    INPUT(["Input Data\nstim · resp · conf · n_ratings"])

    subgraph STDPY ["stdpy — Pure Python SDT"]
        direction TB
        CORE["core.py\ncompute_sdt_resp\ntrials_to_counts"]
        MEASURES["Metacognitive Measures\nfit_meta_d_mle · compute_type2_auc\ncompute_gamma · compute_phi\ncompute_delta_conf\ncompute_meta_uncertainty · compute_meta_noise"]
        COMPUTE_ALL["compute_all_measures\n― 20-measure array ―"]
        CORE --> MEASURES --> COMPUTE_ALL
    end

    subgraph ANALYSIS ["analysis — Inferential Pipeline"]
        direction LR
        BOOTSTRAP["bootstrap_measure\npercentile CI"]
        PERMUTATION["permutation_test\np-value"]
        GROUP["group_summary\ngroup statistics"]
    end

    subgraph EXPERIMENTAL ["Experimental Components"]
        direction LR
        subgraph SDTBAYES ["sdtbayes — Bayesian Estimation  (pip install metasignal[sdtbayes])"]
            direction TB
            subgraph MODELS ["7 Estimation Approaches"]
                direction LR
                HIER["fit_hierarchical_metad\nfit_group_comparison"]
                TWO_STAGE["fit_two_stage_group\nfit_two_stage_comparison"]
                FULL["fit_full_metad\nfit_full_metad_comparison"]
                SUBJECT["fit_subject_level"]
                BETA["fit_beta_auc_group\nfit_beta_auc_comparison"]
                REGR["fit_two_stage_regression\nfit_full_metad_regression"]
                WITHIN["fit_within_subject_comparison"]
            end
            DIAG["diagnostics.py  ·  FitResult\nposterior_summary · convergence_diagnostics\nplot_trace · plot_posterior · plot_forest"]
            MODELS --> DIAG
        end

        subgraph ITMC ["itmc — Information-Theoretic Metacognition"]
            direction TB
            ITMC_FN["meta_I · meta_Ir1 · meta_Ir1_acc · meta_Ir2\nRMI · permtest_meta_I"]
        end
    end

    CLI(["metasignal compute\nCLI"])

    INPUT --> STDPY
    COMPUTE_ALL --> BOOTSTRAP
    COMPUTE_ALL --> PERMUTATION
    COMPUTE_ALL --> GROUP
    COMPUTE_ALL --> CLI
    STDPY --> EXPERIMENTAL

    classDef io fill:#2c3e50,color:#fff,stroke:#1a252f
    classDef measure fill:#e8f4fc,color:#1a5276,stroke:#2980b9
    classDef inference fill:#e9f7ef,color:#1e8449,stroke:#27ae60
    classDef bayes fill:#f5eef8,color:#7d3c98,stroke:#8e44ad
    classDef itmc fill:#fdf1e3,color:#a85b00,stroke:#c77b00
    classDef diag fill:#7d3c98,color:#fff,stroke:#6c3483

    class INPUT,CLI io
    class CORE,MEASURES,COMPUTE_ALL measure
    class BOOTSTRAP,PERMUTATION,GROUP inference
    class HIER,TWO_STAGE,FULL,SUBJECT,BETA,REGR,WITHIN bayes
    class ITMC_FN itmc
    class DIAG diag
```

## Key

| Colour      | Meaning                                                   |
| ----------- | --------------------------------------------------------- |
| Dark        | Input / CLI entry points                                  |
| Blue        | `stdpy` SDT computation layer                             |
| Green       | `analysis` inferential pipeline                           |
| Purple      | `sdtbayes` Bayesian estimation (optional, experimental)   |
| Amber       | `itmc` information-theoretic metacognition (experimental) |
| Solid arrow | Data / control flow                                       |

Rendered as `paper/structure.png` by `scripts/make_structure_figure.py` (matplotlib, not this Mermaid source — kept here as a human-readable outline of the same layout; keep both in sync when the architecture changes).
