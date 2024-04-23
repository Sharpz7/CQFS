from data.DataLoader import TheBooksDatasetLoader
from experiments.baseline_TFIDF import baseline_TFIDF


def main():
    data_loader = TheBooksDatasetLoader()
    ICM_name = 'ICM_all'
    baseline_TFIDF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
