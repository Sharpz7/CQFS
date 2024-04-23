from data.DataLoader import TheBooksDatasetLoader
from experiments.train_CF import train_CF


def main():
    data_loader = TheBooksDatasetLoader()
    train_CF(data_loader)


if __name__ == '__main__':
    main()
