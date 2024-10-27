import pandas as pd
from pathlib import Path

def load_csv_as_df(input_file: str) -> pd.DataFrame:
    __validate_input(input_file)
    return pd.read_csv(input_file)

def __validate_input(input_file: str) -> bool:
    if not input_file:
        raise ValueError("Input file is not provided")
    if not Path(input_file).exists():
        raise FileNotFoundError("Input file does not exist")
    return True

def df_print(df: pd.DataFrame) -> None:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)