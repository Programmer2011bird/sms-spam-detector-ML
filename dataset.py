import pandas as pd


def process_raw_txt(IN_file: str, OUT_file: str):
    file_path: str = IN_file
    df: pd.DataFrame = pd.read_csv(file_path, sep="\t", header=None, names=["label", "text"])
    
    with open(OUT_file, "w+") as file:
        file.write(df.to_csv(index=False))


if __name__ == "__main__":
    process_raw_txt("./data/SMSSpamCollection", "./data/SMSSpam.csv")
