from tinygrad import Tensor, TinyJit, nn
from typing import List, Callable
from sklearn import metrics
import config
import vllm

class ClassificationHead:
    def __init__(self, layers, dropout):
        assert layers[-1] == len(config.CWEs)
        assert layers[0] == 768

        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(Tensor.relu)
            self.layers.append(lambda x: x.dropout(dropout))
    
    def forward(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)
        
class MyModel:
    def __init__(self, llm: vllm.LLM, args):
        self.llm = llm
        self.classifier = ClassificationHead(args.layers, args.dropout)
        self.optimizer = nn.optim.SGD(nn.state.get_parameters(self.classifier), lr=args.lr)
    
    def forward(self, x) -> Tensor:
        embed = self.llm.embed(x, use_tqdm=False)
        embeds = [x.outputs.embedding for x in embed]
        tensor = Tensor(embeds, requires_grad=False)
        out = self.classifier.forward(tensor).softmax()
        return out
    
    # @TinyJit
    @Tensor.train()
    def train(self, x, y) -> Tensor:
        self.optimizer.zero_grad()
        out = self.forward(x)
        loss = out.sparse_categorical_crossentropy(y)
        loss.backward()
        self.optimizer.step()
        return loss

    # @TinyJit
    @Tensor.test()
    def test(self, x, y):
        out = self.forward(x)
        loss = out.sparse_categorical_crossentropy(y)
        f1 = metrics.f1_score(y.numpy(), out.numpy().argmax(axis=1), average="macro") 
        return loss, f1