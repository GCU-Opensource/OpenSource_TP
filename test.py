from datasets import Data as data_handler
from model import MaskDetector as torchModel
import torch


def test(root, face, mask):
    # data handler
    data = data_handler(root, face, mask)

    # get testLoader
    _, te_loader = data.RMFD_dataset_handler()

    # model
    model = torchModel()
    model.load_state_dict(torch.load('model.pth'))

    # test
    print(f'Test total: {len(te_loader.dataset)}')
    with torch.no_grad():
        correct = 0
        total = 0
        for idx, (images, labels) in enumerate(te_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total}%')


def main():
    root = 'FaceMask'
    face = 'AFDB_face_dataset'
    mask = 'AFDB_masked_face_dataset'

    test(root, face, mask)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()
