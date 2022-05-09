import numpy as np
import glob
import PIL
import PIL.Image
import tensorflow as tf
import collections
from sklearn.model_selection import train_test_split
import tensorflow_federated as tff
import random

tf.random.set_seed(42)
np.random.seed(42)

# random a da seed ekle

def roundup(nom, denom):
    return nom//denom + (nom%denom > 0)

def get_stanford_federated_dataset(num_clients=3,dirichlet_parameter=10e15,test_ratio=0.2,
                        size = (32,32),TRAIN_EXAMPLES_PER_LABEL=1000):


    data_dir_parent = "C:/Users/temmuz/Desktop/2022-2/proje/stanford/dataset"
    image_names = [[] for _ in range(11)]
    labels = [[] for _ in range(11)]
    for client_num in range(6):
        #print(client_num)
        data_dir = data_dir_parent+"/"+str(client_num)
        for i in range(11):
            data_dir2 = data_dir+"/"+str(i)+"/*.jpg"
            if len(glob.glob(data_dir2))>0:
                temp_names=glob.glob(data_dir2)
                image_names[i].extend(temp_names)
                labels[i].extend(np.ones(len(temp_names),dtype=int)*i)


    train_image_names = []
    labels_train = []
    test_image_names = []
    labels_test = []
    for i in range(11):
        train, test = train_test_split(image_names[i], test_size=test_ratio, random_state=42)
        test_image_names.extend(test)
        labels_test.extend(np.ones(len(test),dtype=int)*i)
        if len(train) < TRAIN_EXAMPLES_PER_LABEL:
            new_num = roundup(TRAIN_EXAMPLES_PER_LABEL, len(train))
            temp_names = train*new_num
            random.shuffle(temp_names)
            train_image_names.extend(temp_names[:TRAIN_EXAMPLES_PER_LABEL])
            labels_train.extend(np.ones(TRAIN_EXAMPLES_PER_LABEL,dtype=int)*i)
        else:
            temp_names = train
            random.shuffle(temp_names)
            train_image_names.extend(temp_names[:TRAIN_EXAMPLES_PER_LABEL])
            labels_train.extend(np.ones(TRAIN_EXAMPLES_PER_LABEL,dtype=int)*i)


    NUM_CLASSES = 11
    TRAIN_EXAMPLES = TRAIN_EXAMPLES_PER_LABEL*NUM_CLASSES
    train_labels = np.array(labels_train)
    test_labels = np.array(labels_test)
    train_clients = collections.OrderedDict()
    test_clients = collections.OrderedDict()
    train_multinomial_vals = []
    # Each client has a multinomial distribution over classes drawn from a
    # Dirichlet.
    for i in range(num_clients):
        proportion = np.random.dirichlet(dirichlet_parameter *
                                        np.ones(NUM_CLASSES,))
        train_multinomial_vals.append(proportion)

    train_multinomial_vals = np.array(train_multinomial_vals)

    train_example_indices = []
    for k in range(NUM_CLASSES):
        train_label_k = np.where(train_labels == k)[0]
        np.random.shuffle(train_label_k)
        train_example_indices.append(train_label_k)

    train_example_indices = np.array(train_example_indices)

    train_client_samples = [[] for _ in range(num_clients)]
    train_count = np.zeros(NUM_CLASSES).astype(int)

    train_examples_per_client = int(TRAIN_EXAMPLES / num_clients)
    for k in range(num_clients):
        for i in range(train_examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, train_multinomial_vals[k, :]) == 1)[0][0]
            train_client_samples[k].append(
                train_example_indices[sampled_label, train_count[sampled_label]])
            train_count[sampled_label] += 1
            if train_count[sampled_label] == TRAIN_EXAMPLES_PER_LABEL:
                train_multinomial_vals[:, sampled_label] = 0
                train_multinomial_vals = (train_multinomial_vals / train_multinomial_vals.sum(axis=1)[:, None])

    for i in range(num_clients):
        client_name = i
        #x_train = train_images[np.array(train_client_samples[i])]
        #x_train = [train_images[x] for x in np.array(train_client_samples[i])]
        x_train = [tf.keras.preprocessing.image.img_to_array(PIL.Image.open(train_image_names[x]).resize(size))/255
            for x in np.array(train_client_samples[i])]
        y_train = train_labels[np.array(
            train_client_samples[i])].squeeze()
        train_data = collections.OrderedDict(
            (('image', x_train), ('label', y_train)))
        train_clients[client_name] = train_data
        
    x_test = [tf.keras.preprocessing.image.img_to_array(PIL.Image.open(test_image_names[x]).resize(size))/255
            for x in range(len(labels_test))]
    test_data = collections.OrderedDict((('image', x_test), ('label', test_labels)))
    test_clients[0] = test_data

    train_dataset = tff.simulation.FromTensorSlicesClientData(train_clients)
    test_dataset = tff.simulation.FromTensorSlicesClientData(test_clients)

    return train_dataset, test_dataset


def preprocess(dataset, batch_size=20):

    def batch_format_fn(element):
        return collections.OrderedDict(
                x=element['image'],
                y=tf.reshape(element['label'], [-1, 1]))

    return dataset.shuffle(15000, seed=42).batch(batch_size).map(batch_format_fn)