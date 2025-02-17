import numpy as np
import pickle as pk
import numba as nb
import time
import random
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_iris

# --- Global Constants ---
normal_distribution_mean = 0
vectorized_gene_testing = 0
get_all_data_by_class = 0

# CIFAR Constants
data_dir = './cifar/'
data_sets_length = 10000
train_data_sets_amount = 5
train_data_set_basename = 'data_batch_'
greyscale_name = 'gs_'

test_data_set_name = 'test_batch'

cifar_class_amount = 5
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Iris Constants
iris = load_iris()
iris_train_data = iris.data
iris_train_labels = iris.target

iris_ranges = []

# --- Global Variables ---
test_data = {}
train_data = {}
class_amount = 0
data_columns_amount = 0
median = 0
std_dev = 0.5
class_array = np.asarray([], dtype=np.uint8)
class_distribution = {}

current_generation = np.asarray([])
generations_number = 0


# Remove n amount of classes from data
def rem_classes(data, labels, n):
    global class_array
    unique_classes = np.unique(labels)

    if unique_classes.size <= n:
        return

    if class_array.size == 0:
        class_array = np.take(unique_classes, random.sample(range(unique_classes.size - 1), n))

    indexes_to_take = np.asarray([])

    for i in range(class_array.size):
        if i == 0:
            indexes_to_take = np.argwhere(labels == class_array[i])
        else:
            indexes_to_take = np.append(indexes_to_take, np.argwhere(labels == class_array[i]), axis=0)

    indexes_to_take = indexes_to_take.reshape(-1)

    return [np.take(data, indexes_to_take, axis=0), np.take(labels, indexes_to_take)]


# Pickle greyscaled data
def pickle(directory, src):
    # if not Path(directory + greyscale_name + src).is_file():
    file = unpickle(directory + src)

    new_pickle = {
        b'batch_label': file[b'batch_label'],
        b'labels': file[b'labels'],
        b'filenames': file[b'filenames'],
        b'data': np.asarray(greyscale_all_data(file[b'data']), dtype=np.uint8)
    }

    print(new_pickle[b'data'].shape)

    with open(directory + greyscale_name + src, 'wb') as fo:
        pk.dump(new_pickle, fo)

    return new_pickle
    # else:
    #     return {}


# Returns dictionary with unpickled CIFAR data
def unpickle(file):
    with open(file, 'rb') as fo:
        return pk.load(fo, encoding='bytes')


# Turn cifar images to greyscale
def greyscale_all_data(data):
    return np.apply_along_axis(greyscale_image, 1, data)


# Turn array greyscale
def greyscale_image(arr):
    vectorized_greyscale_img = nb.vectorize(greyscale_pixel)
    split_arr = np.split(arr, 3)

    res = vectorized_greyscale_img(split_arr[0], split_arr[1], split_arr[2])

    return res


def greyscale_pixel(r, g, b):
    return int(round(0.2989 * r + 0.5870 * g + 0.1140 * b * 2))


def map_distribution(label, data):
    return {label: data}


# Generate dictionary with labels, data and ids
def map_data(data, labels):
    return np.asarray(list(map(map_single_row, data, labels)))


# Generate dictionary array
def map_single_row(data, label):
    return {'label': label, 'data': data}


# Matrix x Vector multiplication for vectorizing purposes
def matrix_vector_mult(B, A):
    return np.matmul(A, B)


# Get all the data of a certain class
def get_data_by_class(labeled_data, label):
    if labeled_data['label'] == label:
        return labeled_data
    else:
        return


# Gets the index of the probability in the array
def index_of_prob_array(arr, rnd):
    for n in range(arr.size):
        if arr[n] >= rnd:
            return n


# Gets the value of a key in a dictionary list
def get_dict_section(labeled_data, key):
    res = np.array([])
    for i in range(0, labeled_data.size):
        if labeled_data[i] != np.array(None):
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
            loss = loss + max(0, w_row[n] - w_row[np.argwhere(class_array == label)[0, 0]] + 1)
    return loss


# Partial Hinge loss function (doesn't add and normalize)
def partial_hinge_loss(gene_results):
    ordered_labels = train_data['labels']
    lpi = np.asarray([])

    for n in range(0, ordered_labels.size):
        if n == 0:
            lpi = [{'lpi': loss_per_image(gene_results[n], ordered_labels[n]), 'label': ordered_labels[n]}]
        else:
            lpi = np.append(lpi, [{'lpi': loss_per_image(gene_results[n], ordered_labels[n]),
                                   'label': ordered_labels[n]}], axis=0)

    return lpi


# Cross 2 genes
def cross_genes(p1, p2, mutation):
    child_w = np.asarray([])

    is_mutated = False
    if random.uniform(0, 1) <= mutation:
        is_mutated = True

    for n in range(class_amount):
        # Check if this row will be mutated (If loss of either parent's row is good, mutation doesn't happen)
        if is_mutated and random.uniform(0, 1) <= (n + 1) / class_amount and\
                p1['loss-per-class'][n]['class-loss'] >= 0.175 and p2['loss-per-class'][n]['class-loss'] >= 0.175:
            if n == 0:
                child_w = np.asarray([abs(np.random.standard_normal(p1['w'][n].shape) * std_dev) + median])
            else:
                child_w = np.append(child_w,
                                    [abs(np.random.standard_normal(p1['w'][n].shape) * std_dev) + median],
                                    axis=0)
            is_mutated = False
            print('Gene has mutated!')
        else:
            halfsize = int(round(p2['w'][n].size / 2))

            if random.uniform(0, 1) <= 0.5:
                if n == 0:
                    child_w = np.asarray([np.append(p1['w'][n][:halfsize], p2['w'][n][halfsize:], axis=0)])
                else:
                    child_w = np.append(child_w, [np.append(p1['w'][n][:halfsize], p2['w'][n][halfsize:], axis=0)],
                                        axis=0)
            else:
                if n == 0:
                    child_w = np.asarray([np.append(p2['w'][n][:halfsize], p1['w'][n][halfsize:], axis=0)])
                else:
                    child_w = np.append(child_w, [np.append(p2['w'][n][:halfsize], p1['w'][n][halfsize:], axis=0)],
                                        axis=0)

            # if p1['loss-per-class'][n]['class-loss'] >= p2['loss-per-class'][n]['class-loss']:
            #     if n == 0:
            #         child_w = np.asarray([p2['w'][n]])
            #     else:
            #         child_w = np.append(child_w, [p2['w'][n]], axis=0)
            # else:
            #     if n == 0:
            #         child_w = np.asarray([p1['w'][n]])
            #     else:
            #         child_w = np.append(child_w, [p1['w'][n]], axis=0)

    return {'loss': 0,
            'loss-per-class': np.zeros(class_amount),
            'w': child_w}


# Crossover algorithm
def crossover(mutation, children_amount, new_blood):
    global current_generation

    current_generation = np.asarray(sorted(current_generation, key=lambda k: k['loss']))
    loss_array = get_dict_section(current_generation, 'loss')
    max_generation_loss = np.max(loss_array)

    loss_array = (max_generation_loss - loss_array)
    modified_loss_array = loss_array  # **(loss_array/5)  # Remove comment for score inflation/punish
    total_generation_loss = np.sum(modified_loss_array)

    if total_generation_loss == 0:
        fitness_probability_array = np.cumsum(np.zeros(loss_array.size) + (1 / loss_array.size))
    else:
        fitness_probability_array = np.cumsum(modified_loss_array / total_generation_loss)

    new_children = np.asarray([])

    for n in range(int(children_amount)):
        parent_1 = index_of_prob_array(fitness_probability_array, random.uniform(0, 1))
        parent_2 = index_of_prob_array(fitness_probability_array, random.uniform(0, 1))

        while parent_1 == parent_2:
            parent_2 = index_of_prob_array(fitness_probability_array, random.uniform(0, 1))

        # print('Crossing parent ' + str(parent_1) + ' with parent ' + str(parent_2))
        new_child = cross_genes(current_generation[parent_1], current_generation[parent_2], mutation)

        if n == 0:
            new_children = np.asarray([new_child])
        else:
            new_children = np.append(new_children, [new_child])

    new_blood_arr = np.asarray([])
    for i in range(0, new_blood):
        if i == 0:
            new_blood_arr = [create_gene(class_amount, data_columns_amount, mean=median, dev=std_dev)]
        else:
            new_blood_arr = np.append(new_blood_arr,
                                      [create_gene(class_amount, data_columns_amount, mean=median, dev=std_dev)],
                                      axis=0)

    current_generation = np.append(np.append(current_generation[:children_amount-new_blood], new_children), new_blood_arr)


# Get classify results of a gene against all testing data
def test_gene(gene):
    gene_results = np.apply_along_axis(matrix_vector_mult, 1, train_data['data'], gene['w'])
    lpi = partial_hinge_loss(gene_results)
    unique_classes = np.unique(train_data['labels'])

    for n in range(0, unique_classes.size):
        lpi_per_class = get_dict_section(get_all_data_by_class(lpi, unique_classes[n]), 'lpi')
        if n == 0:
            gene['loss-per-class'] = [{'class-loss': np.sum(lpi_per_class) / lpi_per_class.size, 'label': unique_classes[n]}]
        else:
            gene['loss-per-class'] = np.append(gene['loss-per-class'],
                                               [{'class-loss': np.sum(lpi_per_class) / lpi_per_class.size,
                                                 'label': unique_classes[n]}],
                                               axis=0)

    gene['loss'] = np.sum(get_dict_section(lpi, 'lpi')) / lpi.size

    return gene


# Classify whole or partial generation
def test_generation(generation_start=-1, generation_end=-1):
    global current_generation

    if (generation_start == -1 and generation_end == -1) or \
            (generation_start <= 1 and generation_end == -1) or \
            (generation_end >= int(current_generation.size)-2 and generation_start == -1):
        current_generation = vectorized_gene_testing(current_generation)
    else:
        if generation_end == -1:
            current_generation = np.append(current_generation[:generation_start],
                                           vectorized_gene_testing(current_generation[generation_start + 1:]),
                                           axis=0)
        elif generation_start == -1:
            current_generation = np.append(vectorized_gene_testing(current_generation[:generation_end - 1]),
                                           current_generation[generation_end:],
                                           axis=0)

    current_generation = np.asarray(sorted(current_generation, key=lambda k: k['loss']))


# Get accuracy gene against all testing data
def gene_accuracy(gene, testing_sample=True):
    if testing_sample:
        gene_results = np.apply_along_axis(matrix_vector_mult, 1, test_data['data'], gene['w'])
        classification_results = np.argmax(gene_results, axis=1)
        bool_array = classification_results == test_data['labels']
    else:
        gene_results = np.apply_along_axis(matrix_vector_mult, 1, train_data['data'], gene['w'])
        classification_results = np.argmax(gene_results, axis=1)
        bool_array = classification_results == train_data['labels']

    return np.where(bool_array == True)[0].size / bool_array.size


# Creates a new gene
def create_gene(rows, columns, value_selection_method='normal', mean=0.0, dev=1.0):
    if value_selection_method == 'normal':
        return {'loss': 0,
                'loss-per-class': np.zeros(class_amount),
                'w': np.asarray(abs(np.random.standard_normal(size=(rows, columns)) * dev) + mean,
                                dtype=np.float32)}
    elif value_selection_method == 'distributed':
        w = np.asarray([np.random.choice(class_distribution[class_array[0]], columns)], dtype=np.float32)

        for n in range(1, class_array.size):
            w = np.append(w, [np.random.choice(class_distribution[class_array[n]], columns)], axis=0)
        return {'loss': 0,
                'loss-per-class': np.zeros(class_amount),
                'w': w}
    else:
        # Fallback to normal distribution
        return {'loss': 0,
                'loss-per-class': np.zeros(class_amount),
                'w': np.asarray(abs(np.random.standard_normal(size=(rows, columns)) * dev) + mean,
                                dtype=np.float32)}


def get_standard_dist_per_class():
    res_dict = {}

    for i in range(class_array.size):
        max_data = np.max(train_data['data'])
        indexes_to_take = np.argwhere(train_data['labels'] == class_array[i]).flatten()
        data_from_class = (np.take(train_data['data'], indexes_to_take, axis=0) / max_data).flatten()

        res_dict[class_array[i]] = data_from_class

    return res_dict


# Initialize variables and constants for better performance but keeping flexibility
def init(population, is_cifar=False, test_data_amount=0, classes_to_remove=5, value_selection_method='normal'):
    global train_data, test_data, class_amount, data_columns_amount, class_array
    global median, std_dev, generations_number, current_generation
    global vectorized_gene_testing, get_all_data_by_class, class_distribution

    if is_cifar:
        cifar_labels = np.asarray([])
        cifar_data = np.asarray([])
        for data_set_index in range(1, train_data_sets_amount + 1):
            temp_dict = unpickle(data_dir + greyscale_name + train_data_set_basename + str(data_set_index))

            if data_set_index == 1:
                cifar_data = temp_dict[b'data']
                cifar_labels = temp_dict[b'labels']
            else:
                cifar_labels = np.append(cifar_labels, temp_dict[b'labels'], axis=0)
                cifar_data = np.append(cifar_data, temp_dict[b'data'], axis=0)

        tmp_test_data = unpickle(data_dir + greyscale_name + test_data_set_name)

        tmp_test_data_2 = rem_classes(np.asarray(tmp_test_data[b'data'], dtype=np.uint8),
                                      np.asarray(tmp_test_data[b'labels'], dtype=np.uint8),
                                      classes_to_remove)

        removed_classes = rem_classes(cifar_data, cifar_labels, classes_to_remove)
        cifar_data = removed_classes[0]
        cifar_labels = np.asarray(removed_classes[1], dtype=np.uint8)

        cifar_data = np.asarray(np.swapaxes(np.append(np.swapaxes(cifar_data, 0, 1),
                                                      [np.zeros(np.ma.size(np.swapaxes(cifar_data, 0, 1), axis=1)) + 1],
                                                      axis=0), 0, 1), dtype=np.uint8)

        train_data['data'] = cifar_data
        train_data['labels'] = cifar_labels
        test_data['labels'] = tmp_test_data_2[1]
        test_data['data'] = np.asarray(np.swapaxes(np.append(np.swapaxes(tmp_test_data_2[0], 0, 1),
                                                   [np.zeros(np.ma.size(np.swapaxes(tmp_test_data_2[0], 0, 1),
                                                                        axis=1)) + 1], axis=0), 0, 1), dtype=np.uint8)

        # median = np.mean(cifar_data, dtype=np.float32)
        # std_dev = np.std(cifar_data, dtype=np.float32)

        class_amount = np.unique(cifar_labels).size
        data_columns_amount = np.ma.size(cifar_data, axis=1)

    else:
        # Add 1s to iris data for bias trick
        global iris_train_data
        iris_train_data = np.swapaxes(np.append(np.swapaxes(iris_train_data, 0, 1),
                                      [np.zeros(np.ma.size(np.swapaxes(iris_train_data, 0, 1), axis=1)) + 1],
                                      axis=0), 0, 1)

        indexes_to_remove = np.random.choice(range(iris_train_labels.size), test_data_amount, replace=False)

        train_data['data'] = np.delete(iris_train_data, indexes_to_remove, axis=0)
        train_data['labels'] = np.delete(iris_train_labels, indexes_to_remove)

        test_data['data'] = np.take(iris_train_data, indexes_to_remove, axis=0)
        test_data['labels'] = np.take(iris_train_labels, indexes_to_remove)

        class_array = np.unique(iris_train_labels)
        class_amount = np.unique(iris_train_labels).size
        data_columns_amount = np.ma.size(iris_train_data, axis=1)

        median = np.mean(iris_train_data, dtype=np.float32)*2
        std_dev = np.std(iris_train_data, dtype=np.float32)*2

    generations_number = 1
    vectorized_gene_testing = np.vectorize(test_gene)
    get_all_data_by_class = np.vectorize(get_data_by_class)
    class_distribution = get_standard_dist_per_class()

    for i in range(0, population):
        if i == 0:
            current_generation = [create_gene(class_amount, data_columns_amount, mean=median, dev=std_dev,
                                              value_selection_method=value_selection_method)]
        else:
            current_generation = np.append(current_generation,
                                           [create_gene(class_amount, data_columns_amount, mean=median, dev=std_dev,
                                            value_selection_method=value_selection_method)],
                                           axis=0)


def genetic_algorithm(population, generations, mutation, children_per_gen,
                      new_blood_per_gen, test_data_amount=0, is_cifar=False, value_selection_method='normal'):
    global generations_number

    init(population, test_data_amount=test_data_amount, is_cifar=is_cifar,
         value_selection_method=value_selection_method)

    test_generation()

    gen_median_loss = np.asarray([np.average(get_dict_section(current_generation, 'loss'))])
    gen_best_gene_loss = np.asarray([current_generation[0]['loss']])
    gene_best_accuracy = np.asarray([gene_accuracy(current_generation[0]) * 100])

    print('Generation ' + str(generations_number) + ' has an average loss of ' + str(gen_median_loss[0]) +
          ' and the best gene has a an accuracy of ' +
          str(gene_best_accuracy[0]) +
          ' with a loss of ' +
          str(gen_best_gene_loss[0]))
    print('Time Elapsed: ' + str(time.process_time() / 60) + 'm')
    print('----------------------------------------------------------')

    while generations_number < generations and gene_best_accuracy[generations_number-1] <= 95:
        crossover(mutation, children_per_gen, new_blood_per_gen)
        test_generation(generation_start=children_per_gen-new_blood_per_gen)

        gen_median_loss = np.append(gen_median_loss, [np.average(get_dict_section(current_generation, 'loss'))])
        gen_best_gene_loss = np.append(gen_best_gene_loss, [current_generation[0]['loss']])
        gene_best_accuracy = np.append(gene_best_accuracy, [gene_accuracy(current_generation[0]) * 100])
        generations_number = generations_number + 1

        print('Generation ' + str(generations_number) + ' has an average loss of ' +
              str(gen_median_loss[generations_number-1]) +
              ' and the best gene has a an accuracy of ' +
              str(gene_best_accuracy[generations_number - 1]) +
              ' with a loss of ' +
              str(gen_best_gene_loss[generations_number-1]))
        print('Time Elapsed: ' + str(time.process_time() / 60) + 'm')
        print('----------------------------------------------------------')

    if generations_number < generations:
        print()
        print('----------------------------------------------')
        print('---------- ALGORITHM HAS CONVERGED! ----------')
        print('----------------------------------------------')

    plt.plot(range(generations_number), gen_median_loss)
    plt.ylabel('Median Loss')
    plt.show()

    plt.plot(range(generations_number), gen_best_gene_loss)
    plt.ylabel('Best Gene Loss')
    plt.show()

    plt.plot(range(generations_number), gene_best_accuracy)
    plt.ylabel('Best Accuracy')
    plt.show()

    print(current_generation[0])
    print()
    print('Gene accuracy is of ' + str(gene_accuracy(current_generation[0]) * 100) + '%')


def main():
    # pickle(data_dir, test_data_set_name)

    test_data_amount = 50

    # Hyper-Parameters
    generations = 5
    population = 20
    mutation_percentage = 0.01
    children_per_gen = int(population / 3)
    new_blood_per_gen = int(population / 3)

    genetic_algorithm(population, generations, mutation_percentage, children_per_gen,
                      new_blood_per_gen, test_data_amount, is_cifar=True, value_selection_method='normal')


main()
