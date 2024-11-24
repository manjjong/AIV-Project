def get_direction(angle: float):
    if 0 <= angle <= 90:
        return -1, -1
    if 90 <= angle <= 180:
        return -1, 1
    if 180 <= angle <= 270:
        return 1, 1
    return 1, -1


def is_same_direction(diff_x, diff_y, dir_x, dir_y):
    # 방향 비교: 부호가 같으면 True, 다르면 False
    same_x = (diff_x >= 0 and dir_x >= 0) or (diff_x <= 0 and dir_x <= 0)
    same_y = (diff_y >= 0 and dir_y >= 0) or (diff_y <= 0 and dir_y <= 0)

    return same_x and same_y


def detect(angles, points):
    i = 0
    points_size = len(angles)
    defects = []

    while i < points_size:
        angle = angles[i]
        dir_x, dir_y = get_direction(angle)

        pixel_length = 1
        curr_point = points[i]
        next_point = points[(i + 1) % points_size]
        prev_point = points[i - 1]

        diff_x = curr_point[0] - next_point[0]
        diff_y = curr_point[1] - next_point[1]

        if not is_same_direction(curr_point[0] - prev_point[0], curr_point[1] - prev_point[1], dir_x, dir_y):
            defects.append(i)

        if (abs(diff_x) == 1 and diff_y == 0) or (diff_x == 0 and abs(diff_y) == 1):
            j = i + 1
            prev_point = curr_point
            while next_point[0] + diff_x == prev_point[0] and next_point[1] + diff_y == prev_point[1] \
                    and j + 1 < points_size:
                prev_point = next_point
                next_point = points[j + 1]
                j += 1
                pixel_length += 1
            i = j - 1
        i += 1

    return defects
