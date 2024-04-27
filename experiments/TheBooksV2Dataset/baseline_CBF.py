from data.DataLoader import TheBooksV2DatasetLoader
from experiments.baseline_CBF import baseline_CBF


def main():
    data_loader = TheBooksV2DatasetLoader()
    ICM_name = 'ICM_books'
    baseline_CBF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
