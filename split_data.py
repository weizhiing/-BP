import os
import random
import shutil

my_train_file_dir = "my_train/"
my_test_file_dir = "my_test/"

if os.path.exists(my_train_file_dir):
    os.removedirs(my_train_file_dir)

if os.path.exists(my_test_file_dir):
    os.removedirs(my_test_file_dir)

os.mkdir(my_train_file_dir)
os.mkdir(my_test_file_dir)
for i in range(1, 13):
    my_train_file_dir = "my_train/" + str(i) + "/"
    my_test_file_dir = "my_test/" + str(i) + "/"
    os.mkdir(my_train_file_dir)
    os.mkdir(my_test_file_dir)
    from_dir = "train/" + str(i) + "/"
    files = os.listdir(from_dir)
    for file in files:
        if random.random() < 0.8:
            shutil.copyfile(from_dir + file, my_train_file_dir + file)
        else:
            shutil.copyfile(from_dir + file, my_test_file_dir + file)
