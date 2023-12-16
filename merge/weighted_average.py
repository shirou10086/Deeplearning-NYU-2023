import pandas as pd
from sklearn.metrics import accuracy_score

def main():
    # Load data
    file_path = './TrainTest/test_dataset.csv'  # Update with the correct file path if necessary
    data = pd.read_csv(file_path)

    # Given accuracies for NLP and CV
    accuracy_nlp = 0.93
    accuracy_cv = 0.95

    # Calculate weights
    weight_nlp = accuracy_nlp / (accuracy_nlp + accuracy_cv)
    weight_cv = accuracy_cv / (accuracy_nlp + accuracy_cv)

    # Calculate weighted average score
    data['WAscores'] = weight_nlp * data['NLPscores'] + weight_cv * data['CVscores']

    # Convert WA scores to binary predictions and calculate accuracy
    y_pred_wa = (data['WAscores'] > 0.5).astype(int)
    accuracy_wa = accuracy_score(data['Label'], y_pred_wa)

    # Print the weighted average accuracy
    print(f"Weighted Average Accuracy: {accuracy_wa}")

    # Save the updated dataframe back to the same CSV file
    data.to_csv(file_path, index=False)

if __name__ == "__main__":
    main()
