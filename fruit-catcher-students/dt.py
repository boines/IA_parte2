import numpy as np

class DecisionTree:

    # constroi recursivamente a árvore de decisão com base nos exemplos e nos rótulos de treino
    def __init__(self, X, y, threshold=1.0, max_depth=None): # Additional optional arguments can be added, but the default value needs to be provided    
        self.is_leaf = False
        self.label = None
        self.attribute = None
        self.children = {}

        # ver se X ou y estão vazios, assumimos q é fruta
        # mais assumir fruta q bomba porque senão podemos estar a negar frutas
        if not X or not y:
            self.is_leaf = True
            self.label = 1
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
            self.label = max(set(y), key=list(y).count) # label do elemento mais comum
            return

        # TODO: ganhos ser menor q o threshold???
        # calcula o ganho de informação de cada atributo e escolhe o atributo com o maior ganho
        # se o ganho for menor que o threshold, criamos uma folha com o rótulo mais comum
        gains = [self._information_gain(X, y, attr) for attr in range(len(X[0]))]
        self.attribute = np.argmax(gains)
        if gains[self.attribute] < threshold:
            self.is_leaf = True
            self.label = max(set(y), key=list(y).count)
            return

        # TODO: perceber
        # p cada valor possível do atributo ele divide os dados x_sub e y_sub e chama recursivamente a árvore
        values = set([x[self.attribute] for x in X])
        for v in values:
            X_sub = [x[:self.attribute] + x[self.attribute+1:] for x in X if x[self.attribute] == v]
            y_sub = [label for x, label in zip(X, y) if x[self.attribute] == v]
            if not X_sub:
                child = DecisionTree([], [max(set(y), key=list(y).count)], threshold, max_depth, depth+1)
            else:
                child = DecisionTree(X_sub, y_sub, threshold, max_depth, depth+1)
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
        # Implement this
        pass


def train_decision_tree(X, y):
    # Replace with your configuration
    return DecisionTree(X, y)






class DecisionTree:
    def __init__(self, X, y, threshold=1.0, max_depth=None, depth=0):
        self.is_leaf = False
        self.label = None
        self.attribute = None
        self.children = {}

        # If all labels are the same, make a leaf node
        if len(set(y)) == 1:
            self.is_leaf = True
            self.label = y[0]
            return

        # If no attributes left or max_depth reached, make a leaf with majority label
        if len(X[0]) == 0 or (max_depth is not None and depth >= max_depth):
            self.is_leaf = True
            self.label = max(set(y), key=list(y).count)
            return

        # Choose the attribute with highest information gain
        gains = [self._information_gain(X, y, attr) for attr in range(len(X[0]))]
        self.attribute = np.argmax(gains)
        if gains[self.attribute] < threshold:
            self.is_leaf = True
            self.label = max(set(y), key=list(y).count)
            return

        # Split and recurse
        values = set([x[self.attribute] for x in X])
        for v in values:
            X_sub = [x[:self.attribute] + x[self.attribute+1:] for x in X if x[self.attribute] == v]
            y_sub = [label for x, label in zip(X, y) if x[self.attribute] == v]
            if not X_sub:
                # If no examples, use majority label
                child = DecisionTree([], [max(set(y), key=list(y).count)], threshold, max_depth, depth+1)
            else:
                child = DecisionTree(X_sub, y_sub, threshold, max_depth, depth+1)
            self.children[v] = child

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

    def predict(self, x):
        if self.is_leaf:
            return self.label
        v = x[self.attribute]
        if v in self.children:
            # Remove used attribute for child
            x_sub = x[:self.attribute] + x[self.attribute+1:]
            return self.children[v].predict(x_sub)
        else:
            return self.label  # fallback to majority label at this node

def train_decision_tree(X, y):
    return DecisionTree(X, y)