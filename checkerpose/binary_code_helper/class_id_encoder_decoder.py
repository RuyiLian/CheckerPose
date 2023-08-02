import math
import numpy as np
import torch 


def RGB_image_to_class_id_image(RGB_image):
    # RGB_image was saved as BGR in opencv
    # the input of this function has to be numpy array, due to the bit shift
    RGB_image = RGB_image.astype(int)
    class_id_B = np.left_shift(RGB_image[:,:,0], 16)
    class_id_G = np.left_shift(RGB_image[:,:,1], 8)
    class_id_R = RGB_image[:,:,2]

    class_id_image = class_id_B + class_id_G + class_id_R
    return class_id_image

def class_code_images_to_class_id_image(class_code_images, class_base=2):
    """
        class_code_images: numpy array HWC
    """

    class_id_image = np.zeros((class_code_images.shape[0], class_code_images.shape[1]))

    codes_length = class_code_images.shape[2]
    for i in range(codes_length):
        class_id_image =  class_id_image + class_code_images[:,:,i] * (class_base**(codes_length - 1 - i))

    return class_id_image

def  class_code_vecs_to_class_id_vec(class_code_vecs, class_base=2):
    """ Args:
            class_code_vecs: numpy array NxC
    """
    class_id_vec = np.zeros(class_code_vecs.shape[0])
    codes_length = class_code_vecs.shape[1]
    for i in range(codes_length):
        class_id_vec = class_id_vec + class_code_vecs[:, i] * (class_base**(codes_length - 1 - i))
    return class_id_vec

def class_code_images_to_class_id_image_torch(class_code_images, class_base=2):
    """
        class_code_images: torch tensor CHW
    """
    
    class_id_image = torch.zeros((class_code_images.shape[1], class_code_images.shape[2]), dtype=torch.float32).to(class_code_images.device)

    codes_length = class_code_images.shape[0]
    for i in range(codes_length):
        class_id_image =  class_id_image + class_code_images[i,:,:] * (class_base**(codes_length - 1 - i))

    return class_id_image


def class_code_images_to_class_id_image_torch_batch(class_code_images, class_base=2):
    """
        class_code_images: torch tensor BCHW
    """
    batch, codes_length, height, width = class_code_images.shape
    class_id_image = torch.zeros((batch, height, width), dtype=torch.float32).to(class_code_images.device)
    for i in range(codes_length):
        class_id_image = class_id_image + class_code_images[:, i, :, :] * (class_base ** (codes_length - 1 - i))
    class_id_image = class_id_image.long()  # convert to LongTensor
    return class_id_image

def class_id_image_to_class_code_images(class_id_image, class_base=2, iteration=8, number_of_class=256):
    """
        class_id_image: 2D numpy array
    """
    if class_base**iteration != number_of_class:
        raise ValueError('this combination of base and itration is not possible')

    iteration = int(iteration)
    class_code_images = np.zeros((class_id_image.shape[0], class_id_image.shape[1], iteration))
    class_id_image = class_id_image.astype(int)

    bit_step = math.log2(class_base)
    
    for i in range(iteration):
        shifted_value_1 = np.right_shift(class_id_image, int(bit_step * (iteration - i -1)))
        shifted_value_2 = np.right_shift(class_id_image, int(bit_step * (iteration - i)))

        temp = shifted_value_1 - shifted_value_2 * (2**bit_step)
        class_code_images[:,:,i] = shifted_value_1 - shifted_value_2 * (2**bit_step)

    return class_code_images


def class_id_vec_to_class_code_vecs(class_id_vec, class_base=2, iteration=8):
    """
        class_id_vec: 1D numpy array
    """
    iteration = int(iteration)
    class_code_vecs = np.zeros((len(class_id_vec), iteration))
    class_id_vec = class_id_vec.astype(int)

    bit_step = math.log2(class_base)
    for i in range(iteration):
        shifted_value_1 = np.right_shift(class_id_vec, int(bit_step * (iteration - i - 1)))
        shifted_value_2 = np.right_shift(class_id_vec, int(bit_step * (iteration - i)))
        class_code_vecs[:, i] = shifted_value_1 - shifted_value_2 * (2 ** bit_step)
    return class_code_vecs


def code_to_id(class_code, class_base=2):
    """
        class_code: 1d array
    """
    id = 0

    codes_length = len(class_code)
    for i in range(codes_length):
        id =  id + class_code[i] * (class_base**(codes_length - 1 - i))

    return id

def str_code_to_id(str_class_code, class_base=2):
    """
        str_class_code: string
    """
    id = 0

    codes_length = len(str_class_code)
    for i in range(codes_length):
        id =  id + int(str_class_code[i]) * (class_base**(codes_length - 1 - i))

    return id

