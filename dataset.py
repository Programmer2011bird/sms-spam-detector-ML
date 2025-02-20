from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext.transforms import VocabTransform
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad
import pandas as pd
import torch
import re


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def process_raw_txt(IN_file: str, OUT_file: str) -> None:
    file_path: str = IN_file
    df: pd.DataFrame = pd.read_csv(file_path, sep="\t", header=None, names=["label", "text"])
    df["text"].apply(clean_text)
    with open(OUT_file, "w+") as file:
        file.write(df.to_csv(index=False))

class SMSSpamDataset(Dataset):
    def __init__(self, IN_file: str) -> None:
        super().__init__()
        self.df = pd.read_csv(IN_file)
        self.tokenizer = get_tokenizer("basic_english")
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]["text"]
        label = self.df.iloc[index]["label"]
        return self.tokenizer(text), label

class ListToTensor(torch.nn.Module):
    def __init__(self, dtype=torch.long):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return torch.tensor(x, dtype=self.dtype)

class TruncateAndPad(torch.nn.Module):
    def __init__(self, max_length, pad_value):
        super().__init__()
        self.max_length = max_length
        self.pad_value = pad_value

    def forward(self, x):
        if x.size(0) > self.max_length:
            x = x[:self.max_length]  # Truncate
        else:
            x = pad(x, (0, self.max_length - x.size(0)), value=self.pad_value)
        
        return x

raw_dataset = SMSSpamDataset("./data/SMSSpam.csv")

def yield_tokens(data_iter):
    for tokens, _ in data_iter:
        yield tokens

vocab = build_vocab_from_iterator(yield_tokens(raw_dataset), min_freq=2, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

max_sentence_length = 50
text_transform = torch.nn.Sequential(
    VocabTransform(vocab=vocab),
    ListToTensor(),  # Use the custom module
    TruncateAndPad(max_sentence_length, vocab["<unk>"])
)

def collate_batch(batch):
    texts, labels = [], []

    for tokens, label in batch:
        texts.append(text_transform(tokens))
        label = 0 if label == "ham" else 1
        labels.append(label)

    return torch.stack(texts), torch.tensor(labels, dtype=torch.long)

def dataset(BATCH_SIZE: int = 32):
    dataloader = DataLoader(raw_dataset, BATCH_SIZE, collate_fn=collate_batch)

    for texts, labels in dataloader:
        print("Batch shape:", texts.shape)  # (batch_size, seq_length)
        print("Labels:", labels)
        break


if __name__ == "__main__":
    # process_raw_txt("./data/SMSSpamCollection", "./data/SMSSpam.csv")
    dataset()
