import numpy as np
import pandas as pd


class Leaf:
    def __init__(self, class_label):
        self.label = class_label

    def predict(self, row):
        return self.label


class Node:
    def __init__(self, feature):
        self.feature = feature
        self.children = {}

    def add_child(self, value, subtree):
        self.children[value] = subtree

    def predict(self, row):
        value = row[self.feature]
        child = self.children[value]
        return child.predict(row)


class DecisionTreeID3:
    def __init__(self, features, target="label"):
        """
        Inputs:
            features - dict mapping each feature name to the values it can take
            target - name of the target field in the input data
        """
        self.features = features
        self.target = target

    @staticmethod
    def _entropy(data):
        """
        Entropy calculation.
        """
        class_probs = data["label"].value_counts(normalize=True)
        return -np.sum(class_probs * np.log2(class_probs))

    def gain(self, data, feature):
        parent_entropy = self._entropy(data)

        weighted_entropy = 0
        for _, subset in data.groupby(feature):
            weighted_entropy += (len(subset) / len(data)) * self._entropy(subset)

        return parent_entropy - weighted_entropy

    def best_gain(self, data, features):
        best_feature = None
        best_score = -np.inf
        for feature in features:
            score = self.gain(data, feature)
            if score > best_score:
                best_score = score
                best_feature = feature
        return best_feature

    def majority_class(self, data):
        """returns majority class of data"""
        return data["label"].value_counts().idxmax()

    def id3(self, data, features):
        if len(data["label"].unique()) == 1:
            return Leaf(data["label"].iloc[0])

        elif not features:
            return Leaf(self.majority_class(data))

        best = self.best_gain(data, features)
        domain = features[best]
        remaining = {f: v for f, v in features.items() if f != best}

        node = Node(feature=best)
        present = dict(tuple(data.groupby(best)))
        for v in domain:
            if v in present:
                node.add_child(v, self.id3(present[v], remaining))
            else:
                node.add_child(v, Leaf(self.majority_class(data)))

        return node

    def fit(self, data):
        """
        Inputs:
            data - Complete data set of the form (x_i, y_i) where x_i is a feature vector, and y_i is a target value for each data point i.
        """
        self.data = pd.DataFrame(data, columns=list(self.features.keys()) + [self.target])
        self.data = self.data.rename(columns={self.target: "label"})
        self.root = self.id3(self.data, self.features)
        return self

    def predict(self, row):
        return self.root.predict(row)

    def plot(self, ax=None, figsize=(10, 6)):
        """
        Visualizes the fitted tree with matplotlib.
        """
        import matplotlib.pyplot as plt

        positions = {}
        leaf_x = [0]

        def layout(node, depth):
            if isinstance(node, Leaf):
                x = leaf_x[0]
                leaf_x[0] += 1
                positions[id(node)] = (x, -depth)
                return x

            xs = [layout(child, depth + 1) for child in node.children.values()]
            x = sum(xs) / len(xs)
            positions[id(node)] = (x, -depth)
            return x

        layout(self.root, 0)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        def draw(node):
            x, y = positions[id(node)]
            if isinstance(node, Leaf):
                ax.annotate(str(node.label), (x, y), ha="center", va="center",
                            bbox=dict(boxstyle="round", fc="lightgreen"))
                return

            ax.annotate(node.feature, (x, y), ha="center", va="center",
                         bbox=dict(boxstyle="round", fc="lightblue"))

            for value, child in node.children.items():
                cx, cy = positions[id(child)]
                ax.plot([x, cx], [y, cy], color="gray", zorder=0)
                ax.annotate(str(value), ((x + cx) / 2, (y + cy) / 2),
                            ha="center", va="center", fontsize=8,
                            bbox=dict(boxstyle="round", fc="white", ec="none"))
                draw(child)

        draw(self.root)
        ax.axis("off")
        return ax
