from torch.nn import LSTM, Embedding, Linear
import torchtext
import dataset
import torch


class SMSSpamDetectorModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.embedding = Embedding(vocab_size, embedding_dim=embedding_dim)
        self.lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, x):
        self.embedded_text = self.embedding(x)
        _, (_, hidden) = self.lstm(self.embedded_text)

        return self.fc(hidden)


dataloader, VOCAB = dataset.dataset()
VOCAB_SIZE = len(VOCAB)
EMBEDDING_DIM = 100
HIDDEN_DIM = 250
OUTPUT_DIM = 1

model = SMSSpamDetectorModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

for texts, labels in dataloader:
    print("Batch shape:", texts.shape)  # (batch_size, seq_length)
    print("Labels:", labels)
    
    out = model(texts)
    print(out)

    break

