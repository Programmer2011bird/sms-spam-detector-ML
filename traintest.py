import torch
from tqdm import tqdm
import dataset
import model


OPTIMIZER = torch.optim.Adam(model.model.parameters(), lr=0.001)
LOSS_FN = torch.nn.BCELoss()

def train_step(EPOCHS: int):
    dataloader, _ = dataset.dataset()
    print("length of the dataset", len(dataloader))
    
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
            
        print("epoch :", epoch)
        print("epoch loss :", loss)

        torch.save(model.model.state_dict(), "model.pth")

def clean_text(text):
    text = dataset.clean_text(text)
    text = list(text)
    text = dataset.text_transform(text)

    return text

def use_step(IN_FILE):
    model.model.load_state_dict(torch.load(IN_FILE, map_location=torch.device("cpu"), weights_only=True))

    INPUT = "Work as a data entry clerk and earn $300/day. Click here to apply: [link]."
    INPUT = clean_text(INPUT)

    with torch.inference_mode():
        out = model.model(INPUT).squeeze(0)
        out = torch.sigmoid(out).squeeze(-1)
        probabilities = torch.sigmoid(out)
        probabilities = torch.round(probabilities)

        print("0 = ham")
        print("1 = spam")
        print(out)
        print("Probabilities:", probabilities)
        print("Probabilities:", torch.argmax(probabilities))



if __name__ == "__main__":
    # train_step(EPOCHS=1)
    use_step("model.pth")

