import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
file_path = './TrainTest/test_dataset.csv'
data = pd.read_csv(file_path)

# Calculate ROC curve and AUC for each score
fpr_nlp, tpr_nlp, _ = roc_curve(data['Label'], data['NLPscores'])
auc_nlp = auc(fpr_nlp, tpr_nlp)

fpr_cv, tpr_cv, _ = roc_curve(data['Label'], data['CVscores'])
auc_cv = auc(fpr_cv, tpr_cv)

fpr_ir, tpr_ir, _ = roc_curve(data['Label'], data['IRscores'])
auc_ir = auc(fpr_ir, tpr_ir)

fpr_wa, tpr_wa, _ = roc_curve(data['Label'], data['WAscores'])
auc_wa = auc(fpr_wa, tpr_wa)

fpr_rf, tpr_rf, _ = roc_curve(data['Label'], data['RFscores'])
auc_rf = auc(fpr_rf, tpr_rf)

# Plotting ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_nlp, tpr_nlp, color='blue', lw=2, label=f'NLP Score ROC curve (area = {auc_nlp:.6f})')
plt.plot(fpr_cv, tpr_cv, color='green', lw=2, label=f'CV Score ROC curve (area = {auc_cv:.6f})')
plt.plot(fpr_ir, tpr_ir, color='orange', lw=2, label=f'IR Score ROC curve (area = {auc_ir:.6f})')
plt.plot(fpr_wa, tpr_wa, color='purple', lw=2, label=f'WA Score ROC curve (area = {auc_wa:.6f})')
plt.plot(fpr_rf, tpr_rf, color='brown', lw=2, label=f'RF Score ROC curve (area = {auc_rf:.6f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
