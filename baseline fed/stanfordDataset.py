import numpy as np
import glob
import PIL
import PIL.Image
import tensorflow as tf
import collections
from sklearn.model_selection import train_test_split
import tensorflow_federated as tff

def get_stanford_federated_dataset(test_ratio=0.2,size=32):
    data_dir_parent = "C:/Users/temmuz/Desktop/2022-2/proje/stanford/dataset"
    client_train_dataset = collections.OrderedDict()
    client_test_dataset = collections.OrderedDict()
    areas = [[0,2,5],[1,3],[4]]
    ratio0 = 0.2
    ratio1 = [1,1,2]
    ratio2 = 0.7
    ratio3 = 2
    ratio4 = 5
    ratio5 = 0.2
    ratio6 = 17
    ratio7 = [10,10,15]
    ratio8 = 5
    ratio9 = [2,2,4]
    ratio10 = [2,2,4]
    for client_num in range(3):
        #print(client_num)
        train_image_names = []
        test_image_names = []
        labels_train = []
        labels_test = []
        for area in areas[client_num]:
            data_dir = data_dir_parent+"/"+str(area)
            for i in range(11):
                data_dir2 = data_dir+"/"+str(i)+"/*.jpg"
                if len(glob.glob(data_dir2))>0:
                    temp_names=glob.glob(data_dir2)
                    train, test = train_test_split(temp_names, test_size=test_ratio, random_state=42)
                    test_image_names.extend(test)
                    labels_test.extend(np.ones(len(test),dtype=int)*i)
                    if i == 0:
                        full_len = len(train)
                        used_len = int(ratio0*full_len)
                        used_names = train[:used_len]
                    elif i == 1:
                        used_names = train*ratio1[client_num]                    
                    elif i == 2:
                        full_len = len(train)
                        used_len = int(ratio2*full_len)
                        used_names = train[:used_len]
                    elif i == 3:
                        used_names = train*ratio3
                    elif i == 4:
                        used_names = train*ratio4
                    elif i == 5:
                        full_len = len(train)
                        used_len = int(ratio5*full_len)
                        used_names = train[:used_len]
                    elif i == 6:
                        used_names = train*ratio6
                    elif i == 7:
                        used_names = train*ratio7[client_num]
                    elif i == 8:
                        used_names = train*ratio8
                    elif i == 9:
                        used_names = train*ratio9[client_num]
                    elif i == 10:
                        used_names = train*ratio10[client_num]
                    else:
                        used_names = train
                    train_image_names.extend(used_names)
                    labels_train.extend(np.ones(len(used_names),dtype=int)*i)
        if size == 32:
            data_train = np.zeros((len(train_image_names),32,32,3))
            data_test = np.zeros((len(test_image_names),32,32,3))
        elif size == 64:
            data_train = np.zeros((len(train_image_names),64,64,3))
            data_test = np.zeros((len(test_image_names),64,64,3))
        for i, name in enumerate(train_image_names):
            img1 = PIL.Image.open(name)
            if size == 32:
                img1 = img1.resize((32,32))
            elif size == 64:
                img1 = img1.resize((64,64))
            data_train[i] = tf.keras.preprocessing.image.img_to_array(img1)/255
        for i, name in enumerate(test_image_names):
            img1 = PIL.Image.open(name)
            if size == 32:
                img1 = img1.resize((32,32))
            elif size == 64:
                img1 = img1.resize((64,64))
            data_test[i] = tf.keras.preprocessing.image.img_to_array(img1)/255
        data_train = collections.OrderedDict((('label', labels_train), ('image', data_train)))
        data_test = collections.OrderedDict((('label', labels_test), ('image', data_test)))
        client_train_dataset[client_num] = data_train
        client_test_dataset[client_num] = data_test
        # del data_train,data_test,train_image_names,test_image_names,labels_train,labels_test
    train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
    test_dataset = tff.simulation.FromTensorSlicesClientData(client_test_dataset)
    return train_dataset, test_dataset



def preprocess(dataset, batch_size=20):

    def batch_format_fn(element):
        return collections.OrderedDict(
                x=element['image'],
                y=tf.reshape(element['label'], [-1, 1]))

    return dataset.shuffle(15000, seed=42).batch(batch_size).map(batch_format_fn)