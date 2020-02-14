import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from random import seed
from random import randrange
from math import sqrt
from math import pi
from math import exp
from sklearn.preprocessing import LabelEncoder


class NaiveBaye:

    def separate_data(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    # Calculate the following for each data point:
    #   - mean
    #   - standard deviation
    #   - count
    def dataset_statistics(self, dataset):
        summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
        del(summaries[-1])
        return summaries

    # Statistics for each row
    def class_statistics(self, dataset):
        separated = self.separate_data(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.dataset_statistics(rows)
        return summaries

    def probability(self, x, mean, stdev):
        if stdev == 0:
            return 1
        denominator = (sqrt(2 * pi) * stdev)
        numarator = exp(-((x - mean)**2 / (2 * stdev**2)))
        return numarator / denominator

    def likelihood(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, count = class_summaries[i]
                probabilities[class_value] *= self.probability(row[i], mean, stdev)
        return probabilities

    def predict(self, summaries, row):
        probabilities = self.likelihood(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def fit(self, train, test):
        summarize = self.class_statistics(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return(predictions)

    # Calculate accuracy percentage
    def evaluate_acc(self, y, y_hat):
        correct_predictions = 0
        for i in range(len(y)):
            if y[i] == y_hat[i]:
                correct_predictions += 1
        return correct_predictions / float(len(y)) * 100.0

    # K-fold cross validation
    # --------------------------------------------------------------------------------#
    def cross_validation_split(self, dataset, k_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / k_folds)
        for _ in range(k_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Cross validation split
    def evaluate_algorithm(self, dataset, k_folds):
        instances = []
        folds = self.cross_validation_split(dataset, k_folds)
        scores = list()
        instances = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.fit(train_set, test_set)
            actual = [row[-1] for row in fold]
            accuracy = self.evaluate_acc(actual, predicted)
            instances.append(len(test_set) * (len(instances) + 1))
            scores.append(accuracy)
        self.plot_accuracy(scores, instances)
        self.confusion_matrix(actual, predicted)
        return scores

    def confusion_matrix(self, y_hat, y):
        data = {'y_Actual': y,
                'y_Predicted': y_hat}
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        sn.heatmap(confusion_matrix, annot=True)
        plt.show()

    def plot_accuracy(self, scores, instances):
        plt.plot(instances, scores, c='b', label='predicted')
        # plt.scatter(x_data, y_true, c='r', label='true')
        plt.title("Accuracy vs. Instances")
        plt.ylim(0, 100)
        plt.show()
    # --------------------------------------------------------------------------------#

    def main(self):

        # Ionosphere Dataset
        dataframe = pd.read_csv("data/ionosphere.data", header=None)
        array = dataframe.values
        X = array[:, :-1]
        y = array[:, -1:]
        y = LabelEncoder().fit_transform(array[:, -1:].ravel())
        X = X.tolist()
        y = y.tolist()
        array = X
        for i in range(len(X)):
            array[i].append(y[i])
        k_folds = 5
        scores = self.evaluate_algorithm(array, k_folds)
        print('#----------------------------------------#')
        print('Ionosphere data set: ')
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

        # Adult Dataset
        dataframe = pd.read_csv("data/adult.data", header=None)
        dataframe = dataframe.apply(LabelEncoder().fit_transform)
        values = dataframe.values
        array = values.tolist()
        k_folds = 5
        scores = self.evaluate_algorithm(array, k_folds)
        print('#----------------------------------------#')
        print('Adult Dataset: ')
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

        # Breast-Cancer-Wisconsin Dataset
        dataframe = pd.read_csv("data/breast-cancer-wisconsin.data", header=None)
        dataframe = dataframe.replace(to_replace="?", value=np.nan)
        dataframe = dataframe.dropna()
        # dataframe = dataframe.apply(LabelEncoder().fit_transform)
        array = dataframe.values.astype(np.float)
        X = array[:, :-1]
        y = array[:, -1:]
        y = LabelEncoder().fit_transform(array[:, -1:].ravel())
        X = X.tolist()
        y = y.tolist()
        array = X
        for i in range(len(X)):
            array[i].append(y[i])
        k_folds = 5
        scores = self.evaluate_algorithm(array, k_folds)
        print('#----------------------------------------#')
        print('Breast-Cancer Dataset: ')
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

        # Wine Dataset
        dataframe = pd.read_csv("data/winequality-red.csv", sep=';', header=None, skiprows=1)
        values = dataframe.values
        array = values.tolist()
        k_folds = 5
        scores = self.evaluate_algorithm(array, k_folds)
        print('#----------------------------------------#')
        print('Wine Dataset: ')
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


if __name__ == '__main__':
    nm = NaiveBaye()
    nm.main()
