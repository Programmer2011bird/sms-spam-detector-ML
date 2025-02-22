import torch
from tqdm import tqdm
import dataset
import model


OPTIMIZER = torch.optim.Adam(model.model.parameters(), lr=0.001)
LOSS_FN = torch.nn.BCELoss()

def train_step(EPOCHS: int):
    dataloader, _ = dataset.dataset()
    
    for epoch in tqdm(range(EPOCHS)):
        OPTIMIZER.zero_grad()
        model.model.train()
        loss = 0

        for batch, (texts, labels) in enumerate(dataloader):
            predictions = model.model(texts).squeeze(0)
            predictions = torch.sigmoid(predictions).squeeze(-1)

            LOSS = LOSS_FN(predictions, labels.float())
            LOSS.backward()
            loss += LOSS.item()
            OPTIMIZER.step()
            
            print("batch :", batch)

        print("epoch :", epoch)
        print("epoch loss :", loss)

        torch.save(model.model.state_dict(), "model.pth")


if __name__ == "__main__":
    train_step(EPOCHS=1)
