# ANOVA Package

::: daspi.anova
    options:
        members: no

## Module reference

| Module | Contents |
| -------- | ---------- |
| [Linear & Gage Models](linear-model.md) | `LinearModel`, `GageStudyModel`, `GageRnRModel` |
| [Gage Study Model](gage-study-model.md) | `GageStudyModel` — MSA Type-1 with GUM uncertainty budget |
| [Gage R&R Model](gage-rnr-model.md) | `GageRnRModel` — crossed Gage R&R variance components |

## Utility functions

::: daspi.anova.convert
    options:
        members:
            - get_term_name
            - frames_to_html

::: daspi.anova.tables
    options:
        members:
            - uniques
            - terms_effect
            - variance_inflation_factor
            - anova_table
            - terms_probability
