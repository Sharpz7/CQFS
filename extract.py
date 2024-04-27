import os

# Define the base directory where the folders are located
base_dir = "./results/TheBooksV2Dataset/ICM_books/RP3betaRecommender"


# Function to extract the highest precision from a given file
def extract_highest_precision(file_path):
    highest_precision = 0
    try:
        with open(file_path, 'r') as file:
            # only look at the last 5 lines
            file = file.readlines()[-5:]
            for line in file:
                if "PRECISION:" in line and "CUTOFF: " in line:
                    precision_start = line.find("PRECISION:") + 10
                    precision_end = line.find(",", precision_start)
                    precision = float(line[precision_start:precision_end])
                    highest_precision = max(highest_precision, precision)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return highest_precision

# Dictionary to store the highest precision from each 'p' value
precision_results = {}

# Iterate through each subdirectory and process the relevant file
for subdirectory in os.listdir(base_dir):
    # ensure it is a subdirectory and not a file
    if not os.path.isdir(os.path.join(base_dir, subdirectory)):
        continue
    # if directory does not contain the string 'p' then skip
    if 'p' not in subdirectory:
        continue

    # Extract the 'p' value from the subdirectory name
    p_value = subdirectory.split('p')[1]
    file_path = os.path.join(base_dir, subdirectory, "cqfs_simulated_annealing_dwave", "ItemKNNCBFRecommender", "ItemKNNCBFRecommender_ICM_books_cosine_SearchBayesianSkopt.txt")
    precision = extract_highest_precision(file_path)

    # Update the dictionary with the highest precision for each 'p' value
    if p_value not in precision_results or precision_results[p_value][1] < precision:
        precision_results[p_value] = (subdirectory, precision)

# Display the best results for each 'p' value
for p, (subdir, prec) in precision_results.items():
    print(f"The best results for p{p} are in subdirectory {subdir} with a precision of {prec}")
