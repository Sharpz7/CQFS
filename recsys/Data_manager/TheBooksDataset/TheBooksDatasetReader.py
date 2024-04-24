"""
The _loadURM_preinitialized_item_id function has been incorporated into the class.

The AVAILABLE_URM and AVAILABLE_ICM lists have been updated to reflect the
available data in the book recommendation dataset.

The IS_IMPLICIT flag has been set to True, assuming that the dataset
uses implicit ratings (e.g., 0 for not read, 1 for read).

The _loadICM_books function has been added to construct
the ICM from the "Books.csv" file.

The code for loading ICM from credits and metadata
files has been removed, as these files are not present in the book recommendation dataset.

The DATASET_SPECIFIC_MAPPER list has been updated to
include item_original_ID_to_title and item_index_to_title,
which are used to map between book IDs and titles.
"""

import ast
import csv
import os
import shutil
import token
import zipfile

import numpy as np
import pandas as pd

from recsys.Base.Recommender_utils import reshapeSparse
from recsys.Data_manager.Booklens._utils_booklens_parser import \
    _loadURM_preinitialized_item_id
from recsys.Data_manager.DataPostprocessing_K_Cores import select_k_cores
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.DataReader_utils import (
    invert_dictionary, merge_ICM, reconcile_mapper_with_removed_tokens,
    remove_features)
from recsys.Data_manager.Dataset import Dataset
from recsys.Data_manager.IncrementalSparseMatrix import \
    IncrementalSparseMatrix_FilterIDs


class TheBooksDatasetReader(DataReader):

    DATASET_URL = (
        "https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset"
    )
    DATASET_SUBFOLDER = "TheBooksDataset/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_all", "ICM_books"]
    DATASET_SPECIFIC_MAPPER = [
        "item_original_ID_to_title",
        "item_index_to_title",
    ]

    IS_IMPLICIT = False  # Assuming the dataset uses implicit ratings (e.g., 0 for not read, 1 for read)

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_zip_file_folder = (
            self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        )
        decompressed_zip_file_folder = (
            self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER
        )

        zipFile_name = "the-books-dataset.zip"

        try:

            dataFile = zipfile.ZipFile(
                compressed_zip_file_folder + zipFile_name
            )

            # users_path = dataFile.extract(
            #     "Users.csv",
            #     path=decompressed_zip_file_folder + "decompressed/",
            # )
            books_path = dataFile.extract(
                "Books.csv",
                path=decompressed_zip_file_folder + "decompressed/",
            )


            URM_PATH = dataFile.extract(
                "Ratings.csv",
                path=decompressed_zip_file_folder + "decompressed/",
            )

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file.")
            self._print(
                "Automatic download not available, please ensure the ZIP data file is in folder {}.".format(
                    compressed_zip_file_folder
                )
            )
            self._print(
                "Data zip file not found or damaged. You may download the data from: {}".format(
                    self.DATASET_URL
                )
            )

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError(
                f"Automatic download not available. {compressed_zip_file_folder + zipFile_name}"
            )

        self.item_original_ID_to_title = {}
        self.item_index_to_title = {}

        # self._print("Loading ICM_users from Books.csv")
        # (
        #     ICM_users,
        #     tokenToFeatureMapper_ICM_users,
        #     self.item_original_ID_to_index,
        # ) = self._loadICM_books(users_path, header=True, if_new_item="add")
        self._print("Loading ICM_books from Books.csv")
        (
            ICM_books,
            tokenToFeatureMapper_ICM_books,
            self.item_original_ID_to_index,
        ) = self._loadICM_books(books_path, header=True, if_new_item="add")

        # n_items = ICM_books.shape[0]

        # ICM_users = reshapeSparse(ICM_users, (n_items, ICM_books.shape[1]))

        self._print("Loading URM from Ratings.csv")
        (
            URM_all,
            self.item_original_ID_to_index,
            self.user_original_ID_to_index,
            URM_timestamp,
        ) = _loadURM_preinitialized_item_id(
            URM_PATH,
            separator=",",
            header=True,
            if_new_user="add",
            if_new_item="ignore",
            item_original_ID_to_index=self.item_original_ID_to_index,
        )

        # Reconcile URM and ICM
        # Keep only items having ICM entries, remove all the others
        self.n_items = ICM_books.shape[0]

        URM_all, removedUsers, removedItems = select_k_cores(
            URM_all, k_value=1, reshape=True
        )
        URM_timestamp, _, _ = select_k_cores(
            URM_timestamp, k_value=1, reshape=True
        )

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(
            self.item_original_ID_to_index, removedItems
        )
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(
            self.user_original_ID_to_index, removedUsers
        )

        # Remove book_ID discarded in previous step
        item_original_ID_to_title_old = self.item_original_ID_to_title.copy()

        for item_id in item_original_ID_to_title_old:

            if item_id not in self.item_original_ID_to_index:
                del self.item_original_ID_to_title[item_id]

        removed_item_mask = np.zeros(self.n_items, dtype=np.bool)
        removed_item_mask[removedItems] = True

        to_preserve_item_mask = np.logical_not(removed_item_mask)

        ICM_books = ICM_books[to_preserve_item_mask, :]
        # ICM_users = ICM_users[to_preserve_item_mask, :]

        self.n_items = ICM_books.shape[0]

        # ICM_all, tokenToFeatureMapper_ICM_all = merge_ICM(
        #     ICM_users,
        #     ICM_books,
        #     tokenToFeatureMapper_ICM_users,
        #     tokenToFeatureMapper_ICM_books,
        # )

        ICM_all = ICM_books
        tokenToFeatureMapper_ICM_all = tokenToFeatureMapper_ICM_books

        self.n_items = ICM_all.shape[0]

        loaded_URM_dict = {"URM_all": URM_all, "URM_timestamp": URM_timestamp}

        loaded_ICM_dict = {"ICM_all": ICM_all, "ICM_books": ICM_books}

        self.loaded_ICM_mapper_dict = {
            "ICM_all": tokenToFeatureMapper_ICM_all,
            "ICM_books": tokenToFeatureMapper_ICM_books,
        }

        additional_data_mapper = {
            "item_original_ID_to_title": self.item_original_ID_to_title,
            "item_index_to_title": self.item_original_ID_to_title,
        }

        loaded_dataset = Dataset(
            dataset_name=self._get_dataset_name(),
            URM_dictionary=loaded_URM_dict,
            ICM_dictionary=loaded_ICM_dict,
            ICM_feature_mapper_dictionary=self.loaded_ICM_mapper_dict,
            UCM_dictionary=None,
            UCM_feature_mapper_dictionary=None,
            user_original_ID_to_index=self.user_original_ID_to_index,
            item_original_ID_to_index=self.item_original_ID_to_index,
            is_implicit=self.IS_IMPLICIT,
            additional_data_mapper=additional_data_mapper,
        )

        self._print("cleaning temporary files")

        shutil.rmtree(
            decompressed_zip_file_folder + "decompressed/", ignore_errors=True
        )

        self._print("loading complete")

        return loaded_dataset

    def _loadICM_books(self, books_path, header=True, if_new_item="add"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(
            preinitialized_col_mapper=None,
            on_new_col="add",
            preinitialized_row_mapper=None,
            on_new_row=if_new_item,
        )

        books_file = open(books_path, "r", encoding="utf8")

        if header:
            books_file.readline()

        parser_books = csv.reader(books_file, delimiter=",", quotechar='"')

        for book_data in parser_books:
            book_id = book_data[0]
            title = book_data[1]
            author = book_data[2]
            year = book_data[3]
            publisher = book_data[4]

            self.item_original_ID_to_title[book_id] = title

            token_list = [
                "title_" + title,
                "author_" + author,
                "year_" + year,
                "publisher_" + publisher,
            ]

            ICM_builder.add_single_row(book_id, token_list, data=True)

        return (
            ICM_builder.get_SparseMatrix(),
            ICM_builder.get_column_token_to_id_mapper(),
            ICM_builder.get_row_token_to_id_mapper(),
        )


    # def _loadICM_users(self, books_path, header=True, if_new_item="add"):

    #     ICM_builder = IncrementalSparseMatrix_FilterIDs(
    #         preinitialized_col_mapper=None,
    #         on_new_col="add",
    #         preinitialized_row_mapper=None,
    #         on_new_row=if_new_item,
    #     )

    #     books_file = open(books_path, "r", encoding="utf8")

    #     if header:
    #         books_file.readline()

    #     parser_books = csv.reader(books_file, delimiter=",", quotechar='"')

    #     for book_data in parser_books:
    #         book_id = book_data[0]
    #         location = book_data[1]
    #         age = book_data[2]

    #         # self.item_original_ID_to_title[book_id] = title

    #         token_list = [
    #             "location_" + location,
    #             "age_" + age,
    #         ]

    #         ICM_builder.add_single_row(book_id, token_list, data=True)

    #     return (
    #         ICM_builder.get_SparseMatrix(),
    #         ICM_builder.get_column_token_to_id_mapper(),
    #         ICM_builder.get_row_token_to_id_mapper(),
    #     )

