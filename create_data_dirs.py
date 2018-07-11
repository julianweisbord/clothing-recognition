'''
Created on July 10th, 2018
author: Julian Weisbord
description: Take Deep fashion dataset and create train and val directories to
                feed to the Keras model.
'''
import os
import sys
import shutil
import glob

DATASET = "./deep_fashion/"
TEST_DATASET = "./img/"
CLASSES = ["Jeans", "Sweatpants", "Blazer"]
TRAIN_SPLIT = .8

def create_dirs(dataset, create_test=False, train_split=.8):
    '''
    Description: Take all folders in Deep Fashion that contain one of the classes in the
                     folder name and place all those images in one big folder.
    Input: dataset <string> path, create_test <Bool> create a test folder instead of
               train and val, train_split <float> amount of dataset to reserve
               for training vs validation.
    '''
    files = glob.glob(dataset + '*')
    for clothing_cls in CLASSES:
        class_count = 0
        for dir_str in files:
            if clothing_cls.lower() in dir_str.lower():
                clothing_dir = glob.glob(dir_str + '/*')

                train, val = split_data(clothing_dir, TRAIN_SPLIT)
                if create_test:
                    base_dir = "test_inference/"
                    if not os.path.isdir("test_inference/" + clothing_cls):
                        os.makedirs("test_inference/" + clothing_cls)
                else:
                    base_dir = "train"
                    if not os.path.isdir("train/" + clothing_cls):
                        os.makedirs("train/" + clothing_cls)

                for article_count, img_path in enumerate(train):
                    shutil.copy(img_path, base_dir + clothing_cls + "/{}.jpg".format(class_count + article_count))

                if train_split and train_split < 1:
                    if not os.path.isdir("val/" + clothing_cls):
                        os.makedirs("val/" + clothing_cls)
                    for article_count_val, img_path in enumerate(val):
                        shutil.copy(
                                    img_path, "val/" + clothing_cls + "/{}.jpg"
                                    .format(class_count + article_count + article_count_val + 1))

                class_count += len(clothing_dir)
                print("class_count should increase by: ", len(clothing_dir))
                print("class_count: ", class_count)



def split_data(images, split_ratio):
    '''
    Description: Split a folder of image data into train and val datasets
    Input: images <list of strings> list of image paths, split_ratio <float> amount
               of dataset to reserve for training vs validation.

    Return: train <list of strings> data paths, val <list of strings> data paths.
    '''
    train_num = int(split_ratio * len(images))
    train = images[:train_num]
    val = images[train_num:]

    return train, val

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            create_test = True
            create_dirs(TEST_DATASET, create_test, train_split=None)
        else:
            print("Incorrect command line args")
    else:
        create_dirs(DATASET, train_split=TRAIN_SPLIT)
if __name__ == '__main__':
    main()
