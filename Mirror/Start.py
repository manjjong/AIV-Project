import cv2
import natsort
import numpy as np
import stitch
import os
import argparse


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
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    return images, gray_images


# 경계 후보 추출
def extract_candidates(gray_image, side):
    if side == 'left':
        diff = np.diff(gray_image[:, 0].astype(np.int16))
    elif side == 'right':
        diff = np.diff(gray_image[:, -1].astype(np.int16))
    return np.where(np.abs(diff) > 25)[0]


# image0의 비율에 맞게 image1의 크기를 조정합니다.
def resize_with_aspect_ratio(image0, image1, candidate0, candidate1):
    diff0 = candidate0[-1] - candidate0[0]
    diff1 = candidate1[-1] - candidate1[0]
    ratio = diff0 / diff1
    new_image = cv2.resize(image1, (image1.shape[0], int(image1.shape[1] * ratio)), interpolation=cv2.INTER_LINEAR)

    return new_image, ratio


# 스티칭 준비 데이터 생성
def prepare_stitch_data(images, gray_images):
    width_gap_list = [0] * len(images)
    height_diff_list = [0] * len(images)
    new_images = [None] * len(images)
    ratio_list = [1] * len(images)
    new_images[0] = images[0]

    for i in range(len(gray_images) - 1):
        candidate0 = extract_candidates(gray_images[i], 'right')
        candidate1 = extract_candidates(gray_images[i + 1], 'left')

        # 이전 비율 적용
        prev_ratio = ratio_list[i]
        new_candidate0 = int(candidate0[0] * prev_ratio), int(candidate0[-1] * prev_ratio)

        # 이미지 비율 조정
        new_image, ratio = resize_with_aspect_ratio(images[i], images[i + 1], new_candidate0, candidate1)

        # 새로운 후보 생성
        new_candidate1 = int(candidate1[0] * ratio), int(candidate1[-1] * ratio)

        # 폭 및 높이 차이 계산
        width_gap_list[i + 1] = stitch.do(gray_images[i], gray_images[i + 1])
        height_diff_list[i + 1] = new_candidate0[0] - new_candidate1[0]
        new_images[i + 1] = new_image
        ratio_list[i + 1] = ratio

    return new_images, width_gap_list, height_diff_list


# 스티칭 이미지의 크기를 계산합니다.
def calculate_stitch_dimensions(images, width_gap_list):
    width_list = [images[i].shape[1] - width_gap_list[i] for i in range(len(images))]
    stitch_image_width = sum(width_list)
    stitch_image_height = images[0].shape[0]
    return stitch_image_width, stitch_image_height, width_list


# 스티칭 이미지를 생성합니다.
def create_stitched_image(images, new_images, width_gap_list, height_diff_list, width_list, stitch_image_width, stitch_image_height):
    # 큰 도화지 설정
    stitch_image = np.full((stitch_image_height, stitch_image_width, 3), images[0][1, 1], images[0].dtype)

    # 첫 번째 이미지 복사
    stitch_image[:images[0].shape[0], :images[0].shape[1]] = images[0]

    prev_width = images[0].shape[1]
    prev_height_diff = 0

    # 이미지 이어붙이기
    for i in range(1, len(images)):
        height_diff = height_diff_list[i] + prev_height_diff  # 높이 차이
        width_gap = width_gap_list[i]  # 넓이 차이
        curr_width = prev_width + width_list[i]  # 현재 넓이
        image = new_images[i]

        # 배경 색 설정
        stitch_image[prev_width: curr_width, :] = image[0, 0]

        if height_diff >= 0:
            # 높이 차이가 양수일 경우
            stitch_image[height_diff:, prev_width: curr_width] = \
                image[:stitch_image_height - height_diff, width_gap:]
        else:
            # 높이 차이가 음수일 경우
            stitch_image[:stitch_image_height + height_diff, prev_width: curr_width] = \
                image[-height_diff: stitch_image_height, width_gap:]

        prev_width = curr_width
        prev_height_diff = height_diff

    return stitch_image


def main(folder_path):
    # 이미지 경로 확인
    if not os.path.exists(folder_path):
        print("올바른 이미지 경로를 입력하세요.")
        return

    # 이미지 로드
    images, gray_images = load_images(folder_path)

    # 스티칭 데이터 준비
    new_images, width_gap_list, height_diff_list = prepare_stitch_data(images, gray_images)

    # 스티칭 결과 이미지 크기 계산
    stitch_image_width, stitch_image_height, width_list = calculate_stitch_dimensions(images, width_gap_list)

    # 스티칭 이미지 생성
    stitch_image = create_stitched_image(images, new_images, width_gap_list, height_diff_list, width_list,
                                         stitch_image_width, stitch_image_height)

    if not os.path.exists("Result"):
        os.makedirs("Result")
    # 결과 이미지 저장
    cv2.imwrite("Result/result.bmp", stitch_image)


if __name__ == "__main__":
    folder_path = parse_arguments()
    main(folder_path)
