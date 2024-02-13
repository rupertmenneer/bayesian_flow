import torch
import torchvision

class DynamicallyBinarizedMNISTDataLoaders():

    def __init__(self,):
        # Create the dynamically binarized MNIST dataset
        self.train_dataset = DynamicallyBinarizedMNIST(train=True, download=True)
        self.test_dataset = DynamicallyBinarizedMNIST(train=False, download=True)

        # Create data loaders with the collate function
        self.batch_size = 16
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_dynamic_binarize)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_dynamic_binarize)


class DynamicallyBinarizedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root='./data/', train=True, transform=None, target_transform=None, download=False):
        super(DynamicallyBinarizedMNIST, self).__init__(root, train=train, transform=transform,
                                                        target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def collate_dynamic_binarize(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that samples a binarization probability for each batch.

    Args:
        batch (list[tuple[torch.Tensor, int]]): list of samples to collate.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: resulting batch.
    """
    images, targets = zip(*batch)
    binarization_probs = torch.rand(len(images))
    binarized_images = []
    for img, prob in zip(images, binarization_probs):
        binarized_img = (img > prob).float()
        binarized_images.append(binarized_img)
    return torch.stack(binarized_images)[:, None, ...].to(torch.int64), torch.tensor(targets)