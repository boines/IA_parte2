import numpy as np

class DecisionTree:

    # constroi recursivamente a árvore de decisão com base nos exemplos e nos rótulos de treino
    def __init__(self, X, y, threshold=1.0, max_depth=None, depth=0): # Additional optional arguments can be added, but the default value needs to be provided    
        self.is_leaf = False
        self.label = None
        self.attribute = None
        self.children = {}
        self.majority_label = max(set(y), key=y.count) # para ver se há mais bombas ou frutas até agora

        # ver se X ou y estão vazios, assumimos q é fruta
        # mais assumir fruta q bomba porque senão podemos estar a negar frutas - já não
        if not X or not y:
            self.is_leaf = True
            self.label = -1
            return

        # se todos os rótulos forem iguais, criamos uma folha com esse rótulo
        if len(set(y)) == 1:
            self.is_leaf = True
            self.label = y[0] # label do primeiro exemplo
            return

        # se já não há atributos ou atingimos a profundidade máxima
        # paramos e criamos uma folha com o rótulo mais comum
        if len(X[0]) == 0 or (max_depth is not None and depth >= max_depth):
            self.is_leaf = True
            self.label = -1 if len(set(y)) > 1 else y[0]
            #self.label = max(set(y), key=list(y).count) # label do elemento mais comum
            return

        # TODO: ganhos ser menor q o threshold???
        # calcula o ganho de informação de cada atributo e escolhe o atributo com o maior ganho
        # se o ganho for menor que o threshold, criamos uma folha com o rótulo mais comum
        gains = [self._information_gain(X, y, attr) for attr in range(len(X[0]))]
        #print(f"Gains at depth {depth}: {gains}")
        self.attribute = np.argmax(gains)
        if gains[self.attribute] < threshold: # or len(y) < 2:  # se houver pouca info, assume bomba
            self.is_leaf = True
            self.label = -1 if len(set(y)) > 1 else y[0]
            return

        # TODO: perceber
        # p cada valor possível do atributo ele divide os dados x_reduced e y_sub e chama recursivamente a árvore
        values = set([x[self.attribute] for x in X])
        for v in values:
            x_reduced = [x[:self.attribute] + x[self.attribute+1:] for x in X if x[self.attribute] == v]
            y_sub = [label for x, label in zip(X, y) if x[self.attribute] == v]
            if not x_reduced:
                #child = DecisionTree([], [max(set(y), key=list(y).count)], threshold, max_depth, depth+1) #esta em vez das duas linhas de baixo
                label = -1 if len(set(y)) > 1 else y[0]
                child = DecisionTree([], [label], threshold, max_depth, depth+1)
            else:
                child = DecisionTree(x_reduced, y_sub, threshold, max_depth, depth+1)
            self.children[v] = child

    # calcula a entropia do data set
    def _entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def _information_gain(self, X, y, attr):
        total_entropy = self._entropy(y)
        values, counts = np.unique([x[attr] for x in X], return_counts=True)
        weighted_entropy = 0
        for v, count in zip(values, counts):
            y_sub = [label for x, label in zip(X, y) if x[attr] == v]
            weighted_entropy += (count / len(X)) * self._entropy(y_sub)
        return total_entropy - weighted_entropy


    # classifica um novo objeto segundo os ramos da árvore
    def predict(self, x): # (e.g. x = ['apple', 'green', 'circle'] -> 1 or -1)
        if self.is_leaf:
            return self.label
        v = x[self.attribute]
        if v in self.children:
            x_reduced = x[:self.attribute] + x[self.attribute+1:]
            return self.children[v].predict(x_reduced)
        else:
            return self.majority_label  # fallback mais flexível
            #return -1 # se não houver filho para o valor do atributo, retorna -1 (não é fruta)
            #return self.label

    # só para testar
    def print_tree(self, indent=""):
        if self.is_leaf:
            print(f"{indent}Leaf → Predict: {self.label}")
        else:
            print(f"{indent}Test attribute {self.attribute}")
            for value, child in self.children.items():
                print(f"{indent} └─ If value == {value}:")
                child.print_tree(indent + "     ")

def train_decision_tree(X, y):
    # Replace with your configuration
    return DecisionTree(X, y, threshold=0.0) #threshold=0, para obrigar a arvore a dividir


# testar
import csv
from dt import train_decision_tree

if __name__ == "__main__":
# lê os dados de treino
    def load_dataset(path):
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # skip header
            X, y = [], []
            for row in reader:
                X.append(row[1:-1])          # name, color, format
                y.append(int(row[-1]))       # is_fruit (1 ou -1)
        return X, y

    X_train, y_train = load_dataset("train.csv")
    X_test, y_test = load_dataset("test.csv")

    # treina a árvore
    tree = train_decision_tree(X_train, y_train)

    # print da árvore 
    tree.print_tree()

    # testa a árvore com os dados de teste
    correct = 0
    for x, expected in zip(X_test, y_test):
        pred = tree.predict(x)
        print(f"Input: {x} → Predicted: {pred}, Actual: {expected}")
        if pred == expected:
            correct += 1

    print(f"\nAccuracy: {correct} / {len(X_test)}")