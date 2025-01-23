import argparse
from vllm import LLM
from MyModel import MyModel
from data_loader import load_data
import numpy as np
import config as cfg

def train(args, model, train_data):
    X_train, y_train = train_data
    epochs = args.epochs
    batch_size = args.batch_size
    for epoch in range(epochs):
        losses = []
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            loss = model.train(batch_X, batch_y)
            losses.append(loss.mean().numpy())
        
        # for batch in eval_data:
        #     myModel.eval(batch)

        # myModel.save(f"myModel-{epoch}.pt")

        print(f"Epoch {epoch}: loss={np.sum(losses)/len(losses)}")#, acc={myModel.acc}")

def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--model", default="microsoft/codebert-base", type=str, required=False,
                        help="vLLM model for embeddings.")
    parser.add_argument("--dropout", default=0.1, type=float, required=False,
                        help="Dropout ratio.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Epochs.")
    parser.add_argument("--batch_size", default=10, type=int, required=False,
                        help="Batch size.")
    parser.add_argument("--lr", default=0.01, type=float, required=False,
                        help="Learning rate.")
    parser.add_argument("--do_train", default=True, type=bool, required=False,
                        help="Do train.")
    parser.add_argument("--layers", default=[768, 128, 64, 19], type=list, required=False,
                        help="Sizes of hidden neurons (has to start with 768 and end with 19).")
    parser.add_argument("--data_path", default="../data", type=bool, required=False,
                        help="Path to folder with data.")
    
    args = parser.parse_args()

    llm = LLM(model=args.model, task="embed")
    myModel = MyModel(llm, args)
    if args.do_train:
        train_data = load_data(args.data_path + "train.csv")
        train(args, myModel, train_data)

if __name__ == "__main__":
    main()
