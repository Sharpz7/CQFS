from data.DataLoader import TheBooksDatasetLoader
from experiments.baseline_CBF import baseline_CBF


def main():
    data_loader = TheBooksDatasetLoader()
    ICM_name = 'ICM_all'
    baseline_CBF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
