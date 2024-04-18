import traceback

import pandas as pd

from recsys.Data_manager.IncrementalSparseMatrix import (
    IncrementalSparseMatrix_FilterIDs,
)


def _loadURM_preinitialized_item_id(
    file_path,
    header=False,
    separator=",",
    if_new_user="add",
    if_new_item="ignore",
    item_original_ID_to_index=None,
    user_original_ID_to_index=None,
):

    URM_all_builder = IncrementalSparseMatrix_FilterIDs(
        preinitialized_col_mapper=item_original_ID_to_index,
        on_new_col=if_new_item,
        preinitialized_row_mapper=user_original_ID_to_index,
        on_new_row=if_new_user,
    )

    URM_timestamp_builder = IncrementalSparseMatrix_FilterIDs(
        preinitialized_col_mapper=item_original_ID_to_index,
        on_new_col=if_new_item,
        preinitialized_row_mapper=user_original_ID_to_index,
        on_new_row=if_new_user,
    )

    try:
        if header:
            df_original = pd.read_csv(
                filepath_or_buffer=file_path,
                sep=separator,
                header=0 if header else None,
                usecols=["User-ID", "ISBN", "Book-Rating"],
                dtype={
                    "User-ID": str,
                    "ISBN": str,
                    "Book-Rating": float,
                },
            )
        else:
            df_original = pd.read_csv(
                filepath_or_buffer=file_path,
                sep=separator,
                header=0 if header else None,
                dtype={0: str, 1: str, 2: float},
            )
            df_original.columns = ["User-ID", "ISBN", "Book-Rating"]

    except ValueError as e:
        traceback.print_exc()
        print("===========================\n\n")
        print(f"Error: {e}")
        print(f"Found columns: {df_original.columns.tolist()}")
        exit(1)

    # Remove data with rating non valid
    df_original.drop(
        df_original[df_original["Book-Rating"] == 0.0].index, inplace=True
    )

    user_id_list = df_original["User-ID"].values
    item_id_list = df_original["ISBN"].values
    rating_list = df_original["Book-Rating"].values

    URM_all_builder.add_data_lists(user_id_list, item_id_list, rating_list)
    URM_timestamp_builder.add_data_lists(
        user_id_list, item_id_list, rating_list
    )

    return (
        URM_all_builder.get_SparseMatrix(),
        URM_all_builder.get_column_token_to_id_mapper(),
        URM_all_builder.get_row_token_to_id_mapper(),
        URM_timestamp_builder.get_SparseMatrix(),
    )
