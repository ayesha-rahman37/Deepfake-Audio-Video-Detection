THRESHOLD OPTIMIZATION EXPERIMENT

Purpose
Post-hoc decision-threshold optimization using saved development and test predictions.
No model retraining or model-weight changes are performed.

Input
../results.zip
Expected folders inside ZIP:
- results/reliability_chunks_dev_reliability/ (56 NPZ chunks)
- results/reliability_chunks_reliability/ (56 NPZ chunks; held-out test)

Run
1. Keep threshold_experiment/ beside results.zip.
2. Install dependencies if needed:
   pip install numpy matplotlib scikit-learn
3. Run:
   python threshold_experiment/run_threshold_experiment.py

Protocol
Select the threshold on development predictions by maximizing fake precision while
requiring fake recall >= 0.90. Then evaluate that fixed threshold once on the test set.
Do not tune the threshold using test labels/scores.

Generated outputs
- threshold_experiment_results.json: complete metrics and protocol
- threshold_sweep_test.csv: test sensitivity sweep for thresholds 0.300 to 0.700
- threshold_sensitivity.png: precision/recall/F1 versus threshold
- threshold_confusion_comparison.png: confusion matrix comparison

Key result from the supplied prediction chunks
Selected threshold: 0.5177507401
Test at 0.50: accuracy 68.24%, balanced accuracy 72.80%, fake precision 55.84%,
fake recall 94.72%, fake F1 70.26%, FP 16,754, FN 1,181.
Test at selected threshold: accuracy 71.54%, balanced accuracy 74.64%, fake precision
59.32%, fake recall 89.55%, fake F1 71.37%, FP 13,735, FN 2,337.
False-positive reduction: 18.02% (3,019 fewer), with a 5.17 percentage-point
reduction in fake recall.

Interpretation cautions
- Scores are not established as calibrated probabilities.
- This improves the operating point, not the learned model itself.
- The threshold is chosen on dev and evaluated on test to avoid test-set leakage.
- Report the trade-off; do not claim the optimized threshold is universally superior.
