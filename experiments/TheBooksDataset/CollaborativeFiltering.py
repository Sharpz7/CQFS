import os
import shutil

from data.DataLoader import TheBooksDatasetLoader
from experiments.train_CF import train_CF

# create "results_CF" dir
if not os.path.exists("../../results_CF"):
    os.makedirs("../../results_CF")


def main():
    data_loader = TheBooksDatasetLoader()
    train_CF(data_loader)

    shutil.move("results/TheBooksDataset", "results_CF/TheBooksDataset")



if __name__ == '__main__':
    main()
