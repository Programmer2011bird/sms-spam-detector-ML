import torch
from tqdm import tqdm
import dataset
import model
import matplotlib.pyplot as plt


OPTIMIZER = torch.optim.Adam(model.parameters(), lr=0.000001)
LOSS_FN = torch.nn.BCELoss()

def train_step(EPOCHS: int):
    loss_history = []
    dataloader, _ = dataset()
    print("length of the dataset", len(dataloader))

    for epoch in tqdm(range(EPOCHS)):
        OPTIMIZER.zero_grad()
        model.train().to(DEVICE)
        loss = 0

        for batch, (texts, labels) in enumerate(dataloader):
            predictions = model(texts.to(DEVICE)).to(DEVICE).squeeze(0).to(DEVICE)
            predictions = torch.sigmoid(predictions).squeeze(-1).to(DEVICE)

            LOSS = LOSS_FN(predictions, labels.float().to(DEVICE)).to(DEVICE)
            LOSS.backward()
            loss += LOSS.item() / len(dataloader)
            loss_history.append(LOSS.item() / len(dataloader))
            OPTIMIZER.step()

        print("epoch loss :", loss)

        torch.save(model.state_dict(), "model.pth")

    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

def clean_text(text):
    text = dataset.clean_text(text)
    text = list(text)
    text = dataset.text_transform(text)

    return text

def use_step(IN_FILE):
    model.model.load_state_dict(torch.load(IN_FILE, map_location=torch.device("cpu"), weights_only=True))
    model.model.eval()

    INPUT = "Work as a data entry clerk and earn $300/day. Click here to apply: [link]."
    INPUT = clean_text(INPUT)
    INPUT = torch.unsqueeze(INPUT, 0)

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

