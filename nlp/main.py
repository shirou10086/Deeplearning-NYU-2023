import sys
from data_preparation import load_and_preprocess_data
from train_and_evaluate import train_and_evaluate
from predict import load_model, predict

def main():
    dataset_path = "./dataset/TrainTest/train_dataset.csv"
    model_path = "trained_model"
    num_labels = 2  # 根据你的任务调整这个值

    # 检查命令行参数来确定执行哪个部分
    if len(sys.argv) < 2:
        print("Usage: python main.py [prepare/train/predict]")
        sys.exit(1)

    if sys.argv[1] == "prepare":
        print("Preparing data...")
        load_and_preprocess_data(dataset_path)

    elif sys.argv[1] == "train":
        print("Training model...")
        train_and_evaluate(dataset_path, model_path, num_labels)

    elif sys.argv[1] == "predict":
        if len(sys.argv) < 3:
            print("Usage: python main.py predict 'text to classify'")
            sys.exit(1)
        text_to_classify = sys.argv[2]
        print("Predicting...")
        model, tokenizer = load_model(model_path)
        prediction = predict(text_to_classify, model, tokenizer)
        print("Prediction:", prediction)

    else:
        print("Invalid argument. Use 'prepare', 'train' or 'predict'.")

if __name__ == "__main__":
    main()
