import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import os
from numpy.random import randint,  choice
from torch.utils.data import  Dataset,  ConcatDataset

from random import sample


torch.manual_seed(7)
np.random.seed(0)

def reid_data_prepare(data_list_path, train_dir_path):

    class_img_labels = dict()
    class_cnt = -1
    last_label = -2

    h, w = 224, 224

    transform_train_list = [
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor()]

    transform = transforms.Compose(transform_train_list)
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            if "cuhk01" in data_list_path:
                lbl = int(line[:4])
            else: 
                lbl = int(line.split('_')[0])
            if lbl != last_label:
                class_cnt = class_cnt + 1
                cur_list = list()
                class_img_labels[str(class_cnt)] = cur_list
            last_label = lbl
            img = Image.open(os.path.join(train_dir_path, img))

            img = transform(img)
            class_img_labels[str(class_cnt)].append(img)

    return class_img_labels


def val_data(class_img_labels,class_list,  train=False):

    left_images = list()
    right_images = list()
    binary_label =[]
    for j in range(len(class_list)):
        for epoch in range(2):

            left_label = class_list[j]
            left_size = len(class_img_labels[str(left_label)])

            if (epoch % 2 == 0):
                right_label = np.ones(left_size)*left_label
                same_class = True
            else:
                right_label = sample(list(class_list),left_size)
                same_class = False

            len_left_label_i = left_size
            for i in range(len_left_label_i):
                len_right_label_i = len(class_img_labels[str(int(right_label[i]))])

                left_label_i = i
                right_label_i = choice(len_right_label_i)

                if same_class:
                    while left_label_i == right_label_i:
                        right_label_i = choice(len_right_label_i)
                left_images.append(class_img_labels[str(left_label)][
                                       left_label_i])

                right_images.append(class_img_labels[str(int(right_label[i]))][
                                        right_label_i])

                binary_label.append((left_label == int(right_label[i])).astype(int))

    binary_label = torch.tensor(binary_label, dtype=torch.long).reshape(-1)
     
    print("len val size:", len(left_images))

    return [left_images, right_images], binary_label


def pair_pretrain_on_dataset(source, project_path='./', dataset_parent='./',perc_prior=0.5):
    if source == 'market':
        train_list = project_path + source+ '/train.txt'
        train_dir = dataset_parent + source+ '/bounding_box_train'
        class_count = 750
        test_list = project_path + source+ '/test.txt'
        test_dir = dataset_parent + source+ '/bounding_box_test'
        
    elif source == 'cuhk03':
        train_list = project_path + source+ '/train.txt'
        train_dir = dataset_parent + source+ '/bounding_box_train'
        
        class_count = 767
        test_list = project_path + source+ '/test.txt'
        test_dir = dataset_parent + source+ '/bounding_box_test'

    else:
        train_list = 'unknown'
        train_dir = 'unknown'
        class_count = -1
    
    print("train data :",train_list)
    
    class_img_labels = reid_data_prepare(train_list, train_dir)    
    class_train = class_img_labels
    class_num = len(class_img_labels)
    
    if perc_prior>0:     

        class_val = sample(list(np.arange(len(class_img_labels))), int(len(class_img_labels)*perc_prior))
        class_train = list(set(np.arange(len(class_img_labels))) - set(class_val))

        train =val_data(class_img_labels, class_train, train=True)
        print("load train")
        val = val_data(class_img_labels, class_val, train=False)

    class_test_dict = reid_data_prepare(test_list, test_dir)
    
    class_test = np.arange(len(class_test_dict))
    
    test = val_data(class_test_dict, class_test, train=False)

    if val:   
        print("len train class:", len(train[1]),"len val class:", len(val[1]), "len test class:", len(test[1]))
    else:
        print("len train class:", len(train[1]),"len val class:", 0, "len test class:", len(test[1]))
 
    return train, val, test,class_img_labels, class_val,class_num

class SiameseNetworkDataset(Dataset):

    def __init__(self,data, transform=None, should_invert=True):

        self.pair1 = data[0][0]
        self.pair2 = data[0][1]
        self.label = data[1]
        self.len_data = len(self.label)

    def __getitem__(self, index):
        img0 = self.pair1[index]
        img1 = self.pair2[index]
        label = self.label[index]

        return img0, img1,label

    def __len__(self):
        return self.len_data

    def get_len(self):
        return self.__len__()

def loadbatches(train,val, test, loader_kargs, batch_size, prior=False, perc_train=1.0, perc_prior=0.2):
    """Function to load the batches for the dataset

    Parameters
    ----------
    train : torch dataset object
        train split
    
    test : torch dataset object
        test split 

    loader_kargs : dictionary
        loader arguments
    
    batch_size : int
        size of the batch

    prior : bool
        boolean indicating the use of a learnt prior (e.g. this would be False for a random prior)

    perc_train : float
        percentage of data used for training (set to 1.0 if not intending to do data scarcity experiments)

    perc_prior : float
        percentage of data to use for building the prior (1-perc_prior is used to estimate the risk)

    """

    ntrain = train.get_len()
    ntest = test.get_len()
    print("train data len: ",ntrain,"test data len:",ntest)

    if val:
            concat_data = ConcatDataset([train, val])

            set_bound_1batch = torch.utils.data.DataLoader(
                train, batch_size= ntrain, **loader_kargs)
            set_val_bound = torch.utils.data.DataLoader(
                train, batch_size=batch_size )

            train_loader = torch.utils.data.DataLoader(
                concat_data , batch_size=batch_size )
            
            prior_loader = torch.utils.data.DataLoader(
                val , batch_size=batch_size )

            test_1batch = torch.utils.data.DataLoader(
                test, batch_size=ntest, **loader_kargs)
            test_loader = torch.utils.data.DataLoader(
                test, batch_size=batch_size, **loader_kargs)

    return train_loader, test_loader, prior_loader, set_bound_1batch, test_1batch, set_val_bound
