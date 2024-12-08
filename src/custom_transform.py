import cv2


def remove_hair(image, image_size=(224, 224), filter_size=(5, 5), threshold=5, max_value=255):
    """
    The Squeeze algorithm that was mentioned in an article.
    Remove hair from the image and resize to given size with CUBIC interpolation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blackhat_img = cv2.morphologyEx(gray_image,cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat_img, threshold, max_value, cv2.THRESH_BINARY)

    squeezed_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

    resized_image = cv2.resize(squeezed_image, image_size, interpolation = cv2.INTER_CUBIC)
    return resized_image