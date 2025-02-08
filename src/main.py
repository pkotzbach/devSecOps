import argparse
from vllm import LLM
from MyModel import MyModel
from data_loader import load_data
import numpy as np
import config as cfg

def test(args, model, test_data):
    X_test, y_test = test_data
    batch_size = args.batch_size
    losses = []
    f1s = []
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i:i + batch_size]
        batch_y = y_test[i:i + batch_size]
        loss, f1 = model.test(batch_X, batch_y)
        losses.append(loss.mean().numpy())
        f1s.append(f1.mean())
        # print(f1s[-1])

    return np.sum(losses)/len(losses), np.sum(f1s)/len(f1s)

def train(args, model, train_data, eval_data):
    X_train, y_train = train_data
    epochs = args.epochs
    batch_size = args.batch_size
    for epoch in range(epochs):
        losses = []
        for i in range(0, len(X_train), batch_size):
            # print(f"batch {i//batch_size}/{len(X_train)//batch_size}")
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            loss = model.train(batch_X, batch_y)
            losses.append(loss.mean().numpy())

        if epoch % 10 == 0:
            eval_loss, eval_f1 = test(args, model, eval_data)
            print(f"Epoch {epoch}: loss={np.sum(losses)/len(losses)}, eval_loss={eval_loss}, eval_f1={eval_f1}")
        else:
            print(f"Epoch {epoch}: loss={np.sum(losses)/len(losses)}")

def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--model", default="microsoft/codebert-base", type=str, required=False,
                        help="vLLM model for embeddings.")
    parser.add_argument("--dropout", default=0.1, type=float, required=False,
                        help="Dropout ratio.")
    parser.add_argument("--epochs", default=100, type=int, required=False,
                        help="Epochs.")
    parser.add_argument("--batch_size", default=10, type=int, required=False,
                        help="Batch size.")
    parser.add_argument("--lr", default=0.01, type=float, required=False,
                        help="Learning rate.")
    parser.add_argument("--train", default=1, type=int, required=False,
                        help="Do train.")
    parser.add_argument("--test", default=1, type=int, required=False,
                        help="Do test.")
    parser.add_argument("--layers", default=[768, 128, 64, 19], type=list, required=False,
                        help="Sizes of hidden neurons (has to start with 768 and end with 19).")

    parser.add_argument("--data_folder", default="~/devSecOps/data/", type=str, required=False,
                        help="Path to folder with data.")
    parser.add_argument("--percentage", default=1, type=float, required=False,
                        help="Percentage of datasets (1 = 100%).")
    
    args = parser.parse_args()

    llm = LLM(model=args.model, task="embed")
    myModel = MyModel(llm, args)
    if args.train:
        print("=== TRAIN ===")
        train_data = load_data(args.data_folder + "train.csv", args.batch_size, args.percentage)
        eval_data = load_data(args.data_folder + "eval.csv", args.batch_size, args.percentage)
        train(args, myModel, train_data, eval_data)

    if args.test:
        print("=== TEST ===")
        test_data = load_data(args.data_folder + "test.csv", args.batch_size, args.percentage)
        test_loss, test_f1 = test(args, myModel, test_data)
        print(f"Test loss={test_loss}, Test f1={test_f1}")

if __name__ == "__main__":
    main()
