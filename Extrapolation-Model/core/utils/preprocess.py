__author__ = 'yunbo'

import numpy as np

def reshape_patch(img_tensor, patch_size):
    assert img_tensor.ndim == 5  # Ensure correct input format

    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape

    # Ensure patch_size is valid
    if img_height % patch_size != 0 or img_width % patch_size != 0:
        raise ValueError(f"❌ patch_size={patch_size} must divide both height={img_height} and width={img_width} evenly!")

    # Reshape images into patches
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])  # Ensure correct order

    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                img_height // patch_size,
                                img_width // patch_size,
                                num_channels * patch_size * patch_size])
    print(f"✅ Shape after reshaping: {patch_tensor.shape}")  # Debugging
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size*patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor

