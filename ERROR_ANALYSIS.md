# Full-Test Error Analysis — Reliability Fusion

## Scope

This analysis uses `reliability_mapped_errors.csv`, an error-only export for the full-test reliability-fusion evaluation. It contains misclassified windows only; it does not include correctly classified windows.

## Error counts

| Error type | Misclassified windows | Unique source videos |
|---|---:|---:|
| False positives (real predicted fake) | 16,754 | 12,213 |
| False negatives (fake predicted real) | 1,181 | 1,082 |
| **Total** | **17,935** | **13,094** |

## Score analysis

| Error type | Mean fake score | Median fake score |
|---|---:|---:|
| False positives | 0.5644 | 0.5586 |
| False negatives | 0.4792 | 0.4852 |

## Interpretation

1. False positives (real windows classified as fake) dominate the recorded errors.
2. The mean fake score for false positives is 0.564; many erroneous real windows are therefore relatively close to the 0.5 decision boundary rather than extreme scores.
3. False negatives have a mean fake score of 0.479, also close to the binary decision boundary.
4. These results are consistent with the full-test confusion matrix showing a substantial false-positive burden.
5. The pattern motivates threshold calibration on a held-out validation set, improved video representation learning, and review of audio/video quality and domain-shift effects.

## Limitations

- This export contains errors only. It cannot independently reproduce overall accuracy, precision, recall, F1, or ROC-AUC.
- Window-level rows from one source video may be correlated; counts are not independent-video counts.
- The score is a model output, not a calibrated probability.
- This analysis does not identify the cause of an individual misclassification.

## Recommended follow-up

- Inspect representative false-positive and false-negative source clips.
- Compare modality quality indicators and reliability gates for these errors.
- Tune any decision threshold using validation data only, then evaluate once on the untouched test set.
- Report video-level metrics separately if the project requires video-level decisions.
