import pandas as pd
from pathlib import Path
from src.utils import load_csv_as_df, df_print

def main():
    input_file = "data/gemini.csv"
    df = load_csv_as_df(input_file)
    #print(df.head())
    df_print(df)

if __name__ == "__main__":
    main()