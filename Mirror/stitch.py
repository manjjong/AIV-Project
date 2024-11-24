import cv2


def do(image1, image2):
    image1_width = image1.shape[1]
    template_width = int(image1_width * 0.2)

    max_value = 0
    max_location = None
    index = -1

    for i in range(1, template_width + 1):
        img1 = image1[:, image1_width - i:]
        img2 = image2[:, :i]

        match = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

        if max_val > max_value:
            max_value = max_val
            max_location = max_loc
            index = i

    print(max_value, max_location, index, template_width)
    return index
