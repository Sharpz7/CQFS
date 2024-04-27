# Installation

```bash

# This codebase needs python3.7 to run. This assumes you are running on Ubuntu.
# Otherwise, you will need to compile python from source.
# (Note that this will require loadable-sqlite-extensions)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt install python3.7 python3.7-distutils python3.7-venv -y
sudo apt install gcc python3.7-dev -y

# Setup the Environment
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Compile Cython code
cd recsys
source .venv/bin/activate
python run_compile_all_cython.py

# Download Books.csv and Results.csv from
# https://www.kaggle.com/datasets/saurabhbagchi/books-dataset
# and place in ./booksv2

# Create the dataset
cd booksv2
python create.py
cd ..

# Run Data Preprocessing
mkdir ./recsys/Data_manager_offline_datasets/TheBooksV2Dataset
export PYTHONPATH=$PYTHONPATH:$(pwd)
mv ./booksv2/the-booksv2-dataset.zip ./recsys/Data_manager_offline_datasets/TheBooksV2Dataset

python ./data/split_TheBooksV2Dataset.py

# Run the experiments
cd experiments/TheBooksV2Dataset
python CollaborativeFiltering.py
python CQFS.py
python CQFSTrainer.py

# And the baselines
# =============================

# ItemKNN content-based with all the features
python baseline_CBF.py

# ItemKNN content-based with features selected through TF-IDF
python baseline_TFIDF.py

# CFeCBF feature weighting baseline
python baseline_CFW.py

```

You can look at `results.txt` for the raw results, and `extract.py` for how the right results from the hyperparameter tuning were extracted.