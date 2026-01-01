Hyperparameter Experiment Report – Instrunet AI (Multilabel CNN)

Objective:
The goal of this experiment is to analyze the effect of changing a single hyperparameter on the performance of a multilabel instrument recognition model.

Experiment Setup:
The experiment was conducted using the same multilabel IRMAS dataset, same CNN architecture, same train–validation split, same random seed, and same number of epochs.
Only one hyperparameter was modified: the learning rate.

Baseline Model:
Learning rate: 0.001
Loss function: Binary Cross-Entropy
Output activation: Sigmoid
Task: Multilabel classification

Modified Model:
Learning rate: 0.0001
All other settings were kept identical to the baseline model.

Results Comparison:
The modified model achieved better recall and a higher F1-score compared to the baseline model.
The baseline model converged faster but showed less stable learning.
The modified model learned more gradually and captured multiple instrument labels more effectively.

Observations:
Reducing the learning rate improved convergence stability.
Recall increased, indicating better detection of multiple instruments.
Precision showed a slight trade-off but remained acceptable.
Overall performance improved as measured by the F1-score.

Conclusion:
The change in learning rate from 0.001 to 0.0001 resulted in improved multilabel classification performance.
The modified model is more suitable for multilabel instrument recognition and is selected as the final model.