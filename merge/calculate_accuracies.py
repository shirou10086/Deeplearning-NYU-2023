
import pandas as pd
from sklearn.metrics import accuracy_score

def calculate_accuracy(data, score_column, threshold=0.5):
    # Convert scores to binary predictions
    predictions = (data[score_column] > threshold).astype(int)
    # Calculate accuracy
    accuracy = accuracy_score(data['Label'], predictions)
    return accuracy

def main():
    # Load data
    file_path = './TrainTest/test_dataset.csv'  # Update with the correct file path if necessary
    data = pd.read_csv(file_path)

    # Calculate accuracy for NLP and CV scores
    nlp_accuracy = calculate_accuracy(data, 'NLPscores')
    cv_accuracy = calculate_accuracy(data, 'CVscores')

    # Print the results
    print(f"NLP Accuracy: {nlp_accuracy:.2f}")
    print(f"CV Accuracy: {cv_accuracy:.2f}")

if __name__ == "__main__":
    main()
