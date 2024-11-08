from PIL import Image
ImageLike = Image.Image
from typing import List, Callable

def merge_images_horizontally(image_list: ImageLike):
    """
    Merge list of images horizontally.
    """
    widths, heights = list(), list()
    for img in image_list:
        widths.append(img.size[0])
        heights.append(img.size[1])
    
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in image_list:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_image

def merge_images_vertically(image_list: ImageLike):
    """
    Merge list of images vertically.
    """
    widths, heights = list(), list()
    for img in image_list:
        widths.append(img.size[0])
        heights.append(img.size[1])
    
    max_width = max(widths)
    total_height = sum(heights)

    new_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in image_list:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height

    return new_image

def adjust_size(img: Image.Image, target_max_length: int=1152):
    """
    Rescale the image to meet the max length limit.
    """
    w, h = img.size
    max_ratio = max(img.size[0] / target_max_length, img.size[1] / target_max_length)

    return img.resize((int(w / max_ratio), int(h / max_ratio)))

def zip_images(images: ImageLike, block_size: int=4, method: str="adaptive", pad: bool=False) -> List[Image.Image]:
    """
    Compress the images by a unit of `block_size`. When len(images) < `block_size`, do not merge.
    """
    assert method in ['adaptive', 'vertical', 'horizontal']
    merge_method: Callable[[ImageLike], Image.Image] = None
    base_size = images[0].size
    images = [images[0]] + [img.resize(images[0].size) for img in images[1:]]
    if pad:
        target_len = ((len(images) - 1) // block_size + 1) * block_size
        res = target_len - len(images)
        for i in range(res):
            images.append(Image.new('RGB', images[0].size, color='white'))

    if method == 'vertical':
        merge_method = merge_images_vertically
    elif method == 'horizontal':
        merge_method = merge_images_horizontally
    elif method == 'adaptive':
        if base_size[0] < base_size[1]:
            merge_method = merge_images_horizontally
        else:
            merge_method = merge_images_vertically
    new_images = []
    if len(images) >= block_size:
        for i in range(0, len(images), block_size):
            start = i
            end = i + block_size
            cat = merge_method(images[start:end])
            cat = adjust_size(cat)
            new_images.append(cat)
    else:
        new_images = images
    
    return new_images