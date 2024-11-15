import torchvision.transforms.v2 as T
import PIL

class Augmentations:
    "Re-Exports some augmentations we are using"
    cutout = T.RandomErasing
    translate_and_rotate = T.RandomAffine
    channel_swap = T.RandomChannelPermutation
    flip = T.RandomHorizontalFlip
    grayscale = T.RandomGrayscale

DEFAULT_TRANSFORMS = [
    T.RandomAffine(30,(.1,.1), (1, 1.5)),
    T.RandomChannelPermutation(),
    T.RandomHorizontalFlip(0.5),
    T.RandomGrayscale(0.2),
    T.RandomErasing(0.6),
]

def augment(image, label, transforms = DEFAULT_TRANSFORMS):
    """Will apply the probabilistic transforms to the image and label, guaranteeing that the same things are applied to both, and accepting and returning the label with only one channel for the depth"""
    label = label.convert("RGB") # Need to have same number of channels for some transforms 
    image, label = T.Compose(transforms)(image, label)
    return image, label.convert("L") # Convert back to "grayscale" (just a depth channel)


if __name__ == "__main__":
    img = PIL.Image.open("sample.jpg")
    label = img.convert("L")
    transforms = [
        T.RandomErasing(1),
        T.RandomAffine(10,(.1,.1)),
        T.RandomChannelPermutation(),
        T.RandomHorizontalFlip(1),
        T.RandomGrayscale(0.3),
    ]
    img, label = augment(img, label)#, transforms)
    img.save('tmp.jpg')
    label.save('tmp2.jpg')

