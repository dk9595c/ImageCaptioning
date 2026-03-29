import pickle
import random
import re
from PIL import Image
import numpy as np
import collections
from tqdm import tqdm
from typing import Any

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms

from process_text import pad_captions, unk_captions, build_word_dictionary


def preprocess_captions(captions: list[str], window_size: int) -> None:
    for i, caption in enumerate(captions):
        # convert the caption to lowercase and remove all special characters
        caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())

        # keep only pure-alphabetic words longer than one character
        clean_words = [word for word in caption_nopunct.split()
                       if (len(word) > 1) and (word.isalpha())]

        caption_new = ['<start>'] + clean_words[:window_size - 1] + ['<end>']
        captions[i] = caption_new


def get_image_features(image_names: list[str], data_folder: str, vis_subset: int = 100) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extracts 2048-D ResNet-50 GAP feature vectors for each image.
    Uses torchvision's ResNet-50 (pretrained on ImageNet).
    """
    resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    # remove avgpool and fc so the model outputs spatial feature maps
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2])
    feature_extractor.eval()

    # imagenet normalisation
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    gap = torch.nn.AdaptiveAvgPool2d((1, 1))  # global average pooling → [B, 2048, 1, 1]

    image_features = []
    vis_images = []

    pbar = tqdm(image_names)
    with torch.no_grad():
        for i, image_name in enumerate(pbar):
            img_path = f'{data_folder}/Images/{image_name}'
            pbar.set_description(
                f"[({i+1}/{len(image_names)})] Processing '{img_path}' into 2048-D ResNet GAP Vector"
            )
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb.resize((224, 224)))
                img_tensor = preprocess(img_rgb).unsqueeze(0)   # [1, 3, 224, 224]

            feat_map = feature_extractor(img_tensor)            # [1, 2048, 7, 7]
            feat_vec = gap(feat_map).squeeze(-1).squeeze(-1)    # [1, 2048]
            image_features.append(feat_vec[0].numpy())          # [2048]

            if i < vis_subset:
                vis_images.append(img_array)

    print()
    return image_features, vis_images


def load_data(data_folder: str) -> dict[str, Any]:
    """
    Pre-processes the Flickr 8k dataset into the data.p file consumed by the
    assignment.  You do not need to call this method directly.
    """
    text_file_path = f'{data_folder}/captions.txt'

    with open(text_file_path) as file:
        examples = file.read().splitlines()[1:]

    image_names_to_captions = {}
    for example in examples:
        img_name, caption = example.split(',', 1)
        image_names_to_captions[img_name] = (
            image_names_to_captions.get(img_name, []) + [caption]
        )

    shuffled_images = list(image_names_to_captions.keys())
    random.seed(0)
    random.shuffle(shuffled_images)
    test_image_names  = shuffled_images[:1000]
    train_image_names = shuffled_images[1000:]

    def get_all_captions(image_names):
        return [cap for img in image_names for cap in image_names_to_captions[img]]

    train_captions = get_all_captions(train_image_names)
    test_captions  = get_all_captions(test_image_names)

    window_size = 20
    preprocess_captions(train_captions, window_size)
    preprocess_captions(test_captions,  window_size)

    word_count = collections.Counter()
    for caption in train_captions:
        word_count.update(caption)

    # Replace rare words with <unk>
    unk_captions(train_captions, word_count, 50)
    unk_captions(test_captions,  word_count, 50)

    # Pad captions to uniform length
    pad_captions(train_captions, window_size)
    pad_captions(test_captions,  window_size)

    # Convert tokens to integer indices
    word2idx = build_word_dictionary(train_captions, test_captions)

    print("Getting training embeddings")
    train_image_features, train_images = get_image_features(train_image_names, data_folder)
    print("Getting testing embeddings")
    test_image_features,  test_images  = get_image_features(test_image_names,  data_folder)

    return dict(
        train_captions        = np.array(train_captions),
        test_captions         = np.array(test_captions),
        train_image_features  = np.array(train_image_features),
        test_image_features   = np.array(test_image_features),
        train_images          = np.array(train_images),
        test_images           = np.array(test_images),
        word2idx              = word2idx,
        idx2word              = {v: k for k, v in word2idx.items()},
    )


def create_pickle(data_folder: str) -> None:
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')


if __name__ == '__main__':
    data_folder = '../data'
    create_pickle(data_folder)
