import numpy as np
from abc import abstractmethod
from ..utils.metrics import calculate_entropy, calculate_gini_index, calculate_variance
from ..base import BaseModel

class Node:

    def __init__(self):
        self.is_leaf = False
        self.split_condition = None
        self.left = None
        self.right = None
        pass

class LeafNode(Node):
    def __init__(self, y):
        super().__init__()
        self.is_leaf = True
        self.value = np.mean(y) # prediction value
    pass

class SplitCondition(object):
    def __init__(self, split_variable, split_point):
        self.split_variable = split_variable # feature index
        self.split_point = split_point # threshold
    pass

class DecisionTree(BaseModel):
    def __init__(self, is_regression=True):
        self.root = None
        self.is_regression = is_regression
        self.metric = calculate_variance if self.is_regression else calculate_entropy

        # pruning parameters
        self.max_depth = 64
        self.min_leaf_size = 5
        self.min_gain = 0
        print("initialized decision tree")
        pass

    def _getNumClasses(self, y):
        return y.shape[1] if self.is_regression else np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes

    def _split(self, X, y, i, split_point):
        mask = X[:, i] < split_point
        # print("split y, point, mask: ", X[:, i], split_point, mask)
        # print("mask.shape", mask.shape, np.invert(mask).shape)
        return mask, np.invert(mask)


    def _findBestSplit(self, X, y):
        # TODO: prune tree according to depth and number of data
        if X.shape[0] < self.min_leaf_size:
            return None
        information_gain_max = float('-inf') # minimal information gain required to split
        left_best = right_best = None
        split_variable_best = -1
        split_point_best = -1

        current_information = self.metric(y)

        for i, _ in enumerate(X[0]):
            values = set([x[i] for x in X])
            for _, split_point in enumerate(values):
                mask_left, mask_right = self._split(X, y, i, split_point)
                if X.shape[0] in (np.sum(mask_left), np.sum(mask_right)): # one full subset, one empty subset
                    continue
                information_gain = current_information - np.mean(mask_left) * self.metric(y[mask_left]) - np.mean(mask_right) * self.metric(y[mask_right])
                if information_gain > information_gain_max:
                    information_gain_max = information_gain
                    left_best = mask_left
                    split_variable_best = i
                    split_point_best = split_point
                pass
        if information_gain_max > self.min_gain:
            # print("gain: ", information_gain_max)
            right_best = np.invert(left_best)
            return left_best, right_best, split_variable_best, split_point_best
        return None

    # @abstractmethod
    def buildTree(self, X, y):
        # TODO: deal with multivariate categorical variable: 2^(k-1) - 1 possible splits
        result = self._findBestSplit(X, y)
        if result is None:
            root = LeafNode(y)
            return root
        mask, right_best, split_variable_best, split_point_best = result
        root = Node()
        root.split_condition = SplitCondition(split_variable_best, split_point_best)
        root.left = self.buildTree(X[mask], y[mask])
        root.right = self.buildTree(X[np.invert(mask)], y[np.invert(mask)])
        return root

    def train(self, X, y):
        self.root = self.buildTree(X, y)
        pass

    def loss(self, X, y=None):
        pass

    def predict(self, X):
        Y = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            Y[i] = self.predictOne(x)
        return Y

    def predictOne(self, x):
        node = self.root
        while node and not node.is_leaf:
            if x[node.split_condition.split_variable] <= node.split_condition.split_point:
                node = node.left
            else:
                node = node.right
            pass
        # print(node.value)
        return node.value # predict with the terminal node
    pass

class CART(DecisionTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: use gini index?
        self.metric = calculate_variance if self.is_regression else calculate_entropy
        pass
    pass
