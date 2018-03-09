import numpy as np
import pickle as pk
import numba as nb
import scipy
import time
from sklearn.datasets import load_iris

# --- Global Constants ---
normal_distribution_mean = 0

# CIFAR Constants
data_dir = './cifar/'
data_sets_length = 10000
train_data_sets_amount = 5
train_data_set_basename = 'data_batch_'

test_data_set_name = 'test_batch'

cifar_class_amount = 5
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Iris Constants
iris = load_iris()
iris_train_data = iris.data
iris_train_labels = iris.target

iris_ranges = []

# --- Global Variables ---
test_data = []
train_data = []
class_amount = 0
data_columns_amount = 0
median = 0
std_dev = 1

current_generation = np.ndarray([])
generations_number = 0


# Generate dictionary with labels, data and ids
def map_data(data, labels, is_cifar=True):
    if is_cifar:
        return []
    else:
        return np.asarray(list(map(map_single_row, data, labels)))


# Generate dictionary array
def map_single_row(data, label):
    return {'label': label, 'data': data}


# Array multiplication for vectorizing purposes
def matrix_mult(y, x):
    return np.matmul(x, y)


# Get all the data of a certain class
def get_data_by_class(labeled_data, label):
    if labeled_data['label'] == label:
        return labeled_data
    else:
        return


# Gets the value of a key in a dictionary
def get_dict_section(labeled_data, key):
    res = np.array([])
    for i in range(0, labeled_data.size):
        if i == 0:
            res = [labeled_data[i][key]]
        else:
            res = np.append(res, [labeled_data[i][key]], axis=0)
    return res


# Get loss of gene per image
def loss_per_image(w_row, label):
    loss = 0
    for n in range(0, w_row.size):
        if n != label:
            loss = loss + max(0, w_row[n] - w_row[label] + 1)
    return loss


# Partial Hinge loss function (doesn't add and normalize)
def partial_hinge_loss(gene_results):
    ordered_labels = get_dict_section(train_data, 'label')
    lpi = np.ndarray([])

    for n in range(0, ordered_labels.size):
        if n == 1:
            lpi = [{'lpi': loss_per_image(gene_results[n], ordered_labels[n]), 'label': ordered_labels[n]}]
        else:
            lpi = np.append(lpi, [{'lpi': loss_per_image(gene_results[n], ordered_labels[n]),
                                   'label': ordered_labels[n]}], axis=1)

    return lpi


# Get classify results of a gene
def test_gene(gene):
    gene_results = np.apply_along_axis(matrix_mult, 1, get_dict_section(train_data, 'data'), gene['w'])
    lpi = partial_hinge_loss(gene_results)

    gene['loss-per-class'] = partial_hinge_loss(gene_results)
    gene['loss'] = np.sum(gene['loss-per-class']) / gene['loss-per-class'].size

    return gene


# Classify whole generation
def test_generation(generation):
    global current_generation

    vectorized_testing = np.vectorize(test_gene)
    current_generation = vectorized_testing(current_generation)


# Creates a new gene
def create_gene(rows, columns, value_selection_method='normal', mean=0.0, dev=1.0):
    if value_selection_method == 'normal':
        return {'loss': 0,
                'loss-per-class': np.zeros(class_amount),
                'w': abs(np.random.standard_normal(size=(rows, columns)) * dev) + mean }


# Initialize variables and constants for better performance but keeping flexibility
def init(population, is_cifar=False, test_data_amount=0):
    global train_data, test_data, class_amount, data_columns_amount
    global median, std_dev, generations_number, current_generation

    if is_cifar:
        return
    else:
        # Add 1s to iris data for bias trick
        global iris_train_data
        iris_train_data = np.swapaxes(np.append(np.swapaxes(iris_train_data, 0, 1),
                                      [np.zeros(np.ma.size(np.swapaxes(iris_train_data, 0, 1), axis=1)) + 1],
                                      axis=0), 0, 1)

        mapped_data = map_data(iris_train_data, iris_train_labels, is_cifar=False)
        indexes_to_remove = np.random.choice(range(0, mapped_data.size), test_data_amount, replace=False)

        test_data = np.take(mapped_data, indexes_to_remove)
        train_data = np.delete(mapped_data, indexes_to_remove)

        class_amount = np.unique(iris_train_labels).size
        data_columns_amount = np.ma.size(iris_train_data, axis=1)

        median = np.mean(iris_train_data, dtype=np.float32)
        std_dev = np.std(iris_train_data, dtype=np.float32)

    generations_number = 1
    for i in range(0, population):
        if i == 0:
            current_generation = [create_gene(class_amount, data_columns_amount, mean=median, dev=std_dev)]
        else:
            current_generation = np.append(current_generation,
                                           [create_gene(class_amount, data_columns_amount, mean=median, dev=std_dev)],
                                           axis=0)


# def create_normal_distribution_per_class():
#     global data_per_class
#     get_all_data_by_class = np.vectorize(get_data_by_class)
#
#     for i in range(get_dict_section(train_data, 'label').max()):
#         data_per_class = get_all_data_by_class(train_data, 1)
#         data_per_class = data_per_class[data_per_class != np.array(None)]


# Generate new W from between ranges
# def generate_w(ranges, amount):
def main():
    global iris_train_data
    # Hyper-Parameters
    generations = 100
    population = 100
    mutation_percentage = 0.01

    init(population)
    # print('Test Data -----------------------------------')
    # print(test_data)
    # print('Train Data ----------------------------------')
    # print(train_data)
    print('Median --------------------------------------')
    print(median)
    print('Standard Deviation --------------------------')
    print(std_dev)
    print('Rows ----------------------------------------')
    print(class_amount)
    print('Columns -------------------------------------')
    print(data_columns_amount)
    # print('Gen 1 ---------------------------------------')
    # print(current_generation)
    print('Gen 1,0 -------------------------------------')
    print(current_generation[0])
    print('Iris 0 --------------------------------------')
    print(iris_train_data[0])
    print('Matrix Mult ---------------------------------')
    print(np.matmul(current_generation[0]['w'], iris_train_data[0]))
    print('Total Classification Test -------------------')
    # print(get_dict_section(train_data, 'data'))
    # vectorized_testing = np.apply_along_axis(matrix_mult, 1, get_dict_section(train_data, 'data')
    # , current_generation[0]['w'])
    test_generation(current_generation)
    print(current_generation)
    # print(test_gene(current_generation[0])['loss-per-class'].size)
    # print(np.apply_along_axis(lambda x: for n = range(x.size): if n != 0: x[n] = 0, vectorized_testing))

    # print(train_data)
    # print(train_labels)
    # # print(np.amax(train_data, axis=0))
    # mapped_data = map_data(iris_train_data, iris_train_labels, is_cifar=False)
    # print(mapped_data.size)
    #
    # to_remove = np.random.choice(range(150), 25, replace=False)
    # print(to_remove)
    #
    # global test_data
    # test_data = np.take(mapped_data, to_remove)
    # mapped_data = np.delete(mapped_data, to_remove)
    # print('-----------------------')
    # print(test_data)
    # print('-----------------------')
    # print(mapped_data)
    # print('-----------------------')
    # get_all_data_by_class = np.vectorize(get_data_by_class)
    # only_one = get_all_data_by_class(mapped_data, 1)
    # only_one = only_one[only_one != np.array(None)]
    # print(only_one)
    # print(abs(np.random.standard_normal(size=(3, 4))*3)+4)
    # # all_data = np.apply_along_axis(get_dict_section, 1, only_one, 'data')
    # # print(np.unique(get_dict_section(mapped_data, 'label')).size)
    # all_data = get_dict_section(mapped_data, 'label')
    # # print('fjgfcjgvkvjhbjhjhsadfffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
    # print(np.unique(all_data).size)
    # # print(np.swapaxes(all_data, 0, 1))
    # print(create_gene())
    # unique, counts = np.unique(all_data[:, 3], return_counts=True)
    # print('unique: ')
    # print(unique)
    # print('counts: ')
    # print(counts)
    # print(np.sum(counts))
    # amounts = np.sum(counts)
    # probs = np.apply_along_axis(lambda a: a/amounts, 0, counts)
    # print(probs)
    # print(np.cumsum(probs))
    # print(dict(zip(unique, probs)))

    # print(iris_train_data)
    # print('----------------------------------------------------------')
    # print(iris_train_data[:, 0])
    # unique, counts = np.unique(iris_train_data[:, 0], return_counts=True)
    # print(dict(zip(unique, counts)))
    # print('----------------------------------------------------------')
    # print(iris_train_data[:1])
    # print('----------------------------------------------------------')
    # print(iris_train_data[1:])


main()
