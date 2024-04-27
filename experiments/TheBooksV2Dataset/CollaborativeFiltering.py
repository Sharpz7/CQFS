import os
import shutil

from data.DataLoader import TheBooksV2DatasetLoader
from experiments.train_CF import train_CF


def main():
    data_loader = TheBooksV2DatasetLoader()
    train_CF(data_loader)


if __name__ == '__main__':
    main()
