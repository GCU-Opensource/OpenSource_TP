from datasets import Data as data_handler
from model import MaskDetector as torchModel
import torch


def main():
    root = 'FaceMask'
    face = 'AFDB_face_dataset'
    mask = 'AFDB_masked_face_dataset'

    data = data_handler(root, face, mask)
    _, te_loader = data.RMFD_dataset_handler()

    model = torchModel()
    model.load_state_dict(torch.load('model.pth'))

    """
    Show the images using opencv
    """
    import cv2

    for idx, (images, labels) in enumerate(te_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(images)):
            img = images[i].numpy().transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # check the prediction: if mask, draw a rectangle to the face
            title = 'Mask' if predicted[i] == 1 else 'No Mask'
            if predicted[i] == 1:
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 2)

            cv2.imshow(title, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()