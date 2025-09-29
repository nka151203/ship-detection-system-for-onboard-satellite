import numpy as np
import ultralytics

def split_image(img, patch_size=768):
    """
    Split image into patches of size (patch_size x patch_size).
    Returns a list of tuples: (patch, (x_offset, y_offset)).
    """
    H, W, C = img.shape
    patches = []

    n_h = H // patch_size
    n_w = W // patch_size
    rem_h = H % patch_size
    rem_w = W % patch_size

    for i in range(n_h):
        for j in range(n_w):
            y0 = i * patch_size
            x0 = j * patch_size
            patch = img[y0:y0+patch_size, x0:x0+patch_size, :]
            patches.append((patch, (x0, y0)))

    # horizontal redundancy
    if rem_w > 0:
        for i in range(n_h):
            y0 = i * patch_size
            x0 = W - patch_size
            patch = img[y0:y0+patch_size, x0:W, :]
            patches.append((patch, (x0, y0)))

    # vertical redundancy
    if rem_h > 0:
        for j in range(n_w):
            y0 = H - patch_size
            x0 = j * patch_size
            patch = img[y0:H, x0:x0+patch_size, :]
            patches.append((patch, (x0, y0)))

    # corner redundancy
    if rem_h > 0 and rem_w > 0:
        y0 = H - patch_size
        x0 = W - patch_size
        patch = img[y0:H, x0:W, :]
        patches.append((patch, (x0, y0)))

    return patches 



if __name__ == "__main__":
    img = np.random.randint(0, 256, (110, 110, 3), dtype=np.uint8)
    patches = split_image(img, 25)
    print("Số patch:", len(patches))
    print(type(patches[23][0]))
