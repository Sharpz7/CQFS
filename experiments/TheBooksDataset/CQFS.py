# from dwave.system import LeapHybridSampler
# from neal import SimulatedAnnealingSampler
from core.CQFSSampler import CQFSSimulatedAnnealingSampler
from data.DataLoader import TheBooksDatasetLoader
from experiments.run_CQFS import run_CQFS
from recsys.Recommender_import_list import (ItemKNNCFRecommender,
                                            PureSVDItemRecommender,
                                            RP3betaRecommender)

# from core.CQFSSampler import CQFSQBSolvSampler
# from core.CQFSSampler import CQFSQBSolvTabuSampler


def main():
    data_loader = TheBooksDatasetLoader()
    ICM_name = 'ICM_all'

    ##################################################
    # CQFS hyperparameters and settings

    percentages = [40, 60, 80, 95]
    alphas = [1]
    betas = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    combination_strengths = [1, 10, 100, 1000, 10000]

    parameter_product = True

    ##################################################
    # Samplers

    # solver_class = LeapHybridSampler
    # solver_class = SimulatedAnnealingSampler
    solver_class = CQFSSimulatedAnnealingSampler
    # solver_class = CQFSQBSolvSampler
    # solver_class = CQFSQBSolvTabuSampler

    CF_recommender_classes = [
        ItemKNNCFRecommender,
        PureSVDItemRecommender,
        RP3betaRecommender,
    ]

    save_FPMs = False
    save_BQMs = False

    run_CQFS(
        data_loader,
        ICM_name,
        percentages,
        alphas,
        betas,
        combination_strengths,
        solver_class,
        CF_recommender_classes,
        save_FPMs,
        save_BQMs,
        parameter_product,
    )


if __name__ == "__main__":
    main()
