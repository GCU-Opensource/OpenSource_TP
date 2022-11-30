from datasets import Data as data_handler
from model import MaskDetector as torchModel
import torch
import torch.nn as nn


def train(root, face, mask, lr=0.001, epochs=10):
    # data handler
    data = data_handler(root, face, mask)

    # get trainLoader
    tr_loader, _ = data.RMFD_dataset_handler()

    # model
    model = torchModel()

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    total_step = len(tr_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(tr_loader):
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # save model
    torch.save(model.state_dict(), 'model.pth')


def main():
    root = 'FaceMask'
    face = 'AFDB_face_dataset'
    mask = 'AFDB_masked_face_dataset'

    train(root, face, mask)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()
