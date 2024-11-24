import os
import argparse
import cv2
import natsort
import numpy as np
import defects_by_radius
import circle_defects


# 명령줄 인자를 파싱하여 폴더 경로 반환
def parse_arguments():
    parser = argparse.ArgumentParser(description="이미지 처리 프로그램")
    parser.add_argument("-l", "--dir", type=str, required=True, help="이미지가 저장된 폴더 경로")
    args = parser.parse_args()
    return args.dir


# BMP 파일 목록 정렬, 이미지 로드
def load_images(folder_path):
    # BMP 파일 목록 정렬
    bmp_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.bmp')]
    bmp_files = natsort.natsorted(bmp_files)

    # 이미지 읽기 및 그레이스케일 변환
    images = [cv2.imread(folder_path + "\\" + file) for file in bmp_files]
    return bmp_files, images


# 이미지 전처리
def preprocess_image(images, thresh: int):
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    edges = []
    for gray in gray_images:
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=-1)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=-1)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(sobel_normalized, thresh, 255, cv2.THRESH_BINARY)
        edges.append(binary)
    return edges


# 이미지에서 렌즈의 외곽선, 최소 외접원 찾기
def find_lens_contour(images):
    circles = []
    largest_contours = []
    for image in images:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)
        (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
        circles.append((center_x, center_y, radius))
        largest_contours.append(largest_contour)
    return circles, largest_contours


# 외곽선의 각 점에 대해 각도, 거리, 좌표를 계산
def extract_info(contour, center_x, center_y, radius):
    angles = []
    points = []
    distances = []

    for point in contour:
        x, y = point[0]

        # 거리 계산
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # -를 붙인 이유는 이미지 상 아래로 내려갈 수록 + 되기 때문에 보기 편하게 변환 시킴
        angle = np.arctan2(-(y - center_y), x - center_x) * (180 / np.pi)
        if angle < 0:
            angle += 360  # Normalize angle to [0, 360)

        distances.append(distance - radius)
        points.append(point[0])
        angles.append(angle)

    sorted_indices = np.argsort(angles)
    angles = np.array(angles)[sorted_indices]
    points = np.array(points)[sorted_indices]
    distances = np.array(distances)[sorted_indices]

    return angles, points, distances


# 인덱스 기준으로 근접한 값들을 클러스터로 묶음
def cluster_values(index, angles, threshold):
    clusters = []
    current_cluster = [index[0]]

    for value in index[1:]:
        # 이전 값과의 차이가 threshold 이내이면 같은 클러스터에 추가
        if abs(angles[value] - angles[current_cluster[-1]]) <= threshold:
            current_cluster.append(value)
        else:
            clusters.append(current_cluster)
            current_cluster = [value]

    # 마지막 클러스터 추가
    clusters.append(current_cluster)
    return clusters


# 이미지에서 결함 탐지
def detect_defects(images, circles, contours, bmp_file_names=[], file_save=False):
    defects = []

    for i in range(len(images)):
        center_x, center_y, radius = circles[i]
        contour = contours[i]
        # 외각선에 대한 정보 추출
        angles, points, distances = extract_info(contour, center_x, center_y, radius)
        # 원 패턴을 통해 결함 탐지
        c_defects = circle_defects.detect(angles, points)
        # 중심으로부터 반지를 차이를 이용한 결함 탐지
        small_r_defects, wide_r_defects = defects_by_radius.detect(angles, distances)

        # 공통 조건: 차이가 5도 이내
        common_defects = []

        # 원 패턴 결함과 반지름 이용한 넓은 범위의 결함 공통 각도 찾기
        for index in c_defects:
            c_angle = angles[index]
            # 차이가 5도 이내인 c_defects의 인덱스를 찾음
            close_indices = np.where(np.abs(c_angle - angles[wide_r_defects]) <= 5)[0]
            if close_indices.size > 0:
                common_defects.append(index)  # wide_r_defects의 원래 인덱스

        # 공통 결함과 반지름 이용한 좁은 범위의 결함 합치기
        union_defects = sorted(set(list(common_defects) + list(small_r_defects)))
        defects.append(union_defects)

        if file_save:
             save_detect_result(images[i], bmp_file_names[i], angles, points, union_defects)

    return defects


# 결함 위치를 이미지에 표현 후 저장
def save_detect_result(image, file_name, angles, points, defects):
    new_image = image.copy()
    folder_name = "Defective"

    if len(defects) == 0:
        folder_name = "Normal"
    else:
        cluster_value = cluster_values(defects, angles, 10)

        for value in cluster_value:
            cluster_points = points[value]
            x_min, y_min = np.min(np.array(cluster_points), axis=0) - [30, 30]
            x_max, y_max = np.max(np.array(cluster_points), axis=0) + [30, 30]
            cv2.rectangle(new_image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)

    folder_path = f"Result/{folder_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cv2.imwrite(f"{folder_path}/{file_name}", new_image)


def main(folder_path):
    # 이미지 경로 확인
    if not os.path.exists(folder_path):
        print("올바른 이미지 경로를 입력하세요.")
        return

    # 이미지 로드
    bmp_files, images = load_images(folder_path)

    # 이미지 전처리
    edges_images = preprocess_image(images, 30)

    # 렌즈 외각선 탐지
    circles, contours = find_lens_contour(edges_images)

    # 결함 확인 및 이미지 저장
    detect_defects(images, circles, contours, bmp_files, True)


if __name__ == "__main__":
    folder_path = parse_arguments()
    main(folder_path)
