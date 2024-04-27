import os
import shutil

from dwave.system import LeapHybridSampler
from neal import SimulatedAnnealingSampler

from core.CQFSSampler import (CQFSQBSolvSampler, CQFSQBSolvTabuSampler,
                              CQFSSimulatedAnnealingSampler)
from data.DataLoader import TheBooksDatasetLoader
from experiments.train_CQFS import train_CQFS
from recsys.Recommender_import_list import (ItemKNNCFRecommender,
                                            PureSVDItemRecommender,
                                            RP3betaRecommender)

# make "custom_results" dir
if not os.path.exists("../../results_trainer"):
    os.makedirs("../../results_trainer")

def main():
    data_loader = TheBooksDatasetLoader()
    ICM_name = 'ICM_books'

    percentages = [40, 60, 80, 95]
    alphas = [1]
    betas = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    combination_strengths = [1, 10, 100, 1000, 10000]

    solves_classes = [
        ("Simulated Annealing", SimulatedAnnealingSampler),
        ("CQFS Simulated Annealing", CQFSSimulatedAnnealingSampler),
        ("CQFS QBSolv", CQFSQBSolvSampler),
        ("CQFS QBSolv Tabu", CQFSQBSolvTabuSampler),
    ]

    for name, solver_class in solves_classes:
        CF_recommender_classes = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]

        cpu_count_div = 6
        cpu_count_sub = 0

        train_CQFS(data_loader, ICM_name, percentages, alphas, betas, combination_strengths, solver_class,
                CF_recommender_classes, cpu_count_div=cpu_count_div, cpu_count_sub=cpu_count_sub)

        # move folder "results/TheBooksDataset" to "custom_results"
        shutil.move("results/TheBooksDataset", f"../../results_trainer/TheBooksDataset-{name}")


if __name__ == '__main__':
    main()
