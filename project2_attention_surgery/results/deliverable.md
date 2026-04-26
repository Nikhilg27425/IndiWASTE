# Project 2 Deliverable: The Attention Surgery Experiment

## Baseline Accuracies
| Probe | Accuracy |
| --- | ---: |
| clean | 0.1242 |
| texture | 0.1242 |
| shape | 0.1131 |
| spatial | 0.1353 |

## Top-5 Critical Heads per Probe
### Texture (baseline=0.1242)
| Rank | Layer | Head | Drop |
| ---: | ---: | ---: | ---: |
| 1 | 3 | 4 | +0.0200 |
| 2 | 11 | 3 | +0.0200 |
| 3 | 8 | 3 | +0.0177 |
| 4 | 0 | 6 | +0.0155 |
| 5 | 8 | 2 | +0.0155 |

### Shape (baseline=0.1131)
| Rank | Layer | Head | Drop |
| ---: | ---: | ---: | ---: |
| 1 | 10 | 11 | +0.0200 |
| 2 | 11 | 1 | +0.0133 |
| 3 | 3 | 4 | +0.0111 |
| 4 | 6 | 9 | +0.0111 |
| 5 | 7 | 6 | +0.0111 |

### Spatial (baseline=0.1353)
| Rank | Layer | Head | Drop |
| ---: | ---: | ---: | ---: |
| 1 | 8 | 1 | +0.0177 |
| 2 | 0 | 6 | +0.0155 |
| 3 | 2 | 1 | +0.0155 |
| 4 | 5 | 9 | +0.0155 |
| 5 | 7 | 8 | +0.0155 |

## Interpretation
- Large positive drop = head is critical for that probe.
- Texture-critical heads likely encode local colour/frequency patterns.
- Shape-critical heads likely encode global contour structure.
- Spatial-critical heads likely encode positional patch relationships.

## Output Files
`summary.json` · `results.tsv` · `head_importance.json` · `failure_cases.csv`