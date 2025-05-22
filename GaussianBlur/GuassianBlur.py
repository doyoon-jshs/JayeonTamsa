import cv2

# 이미지 파일 경로 입력 (예: 'input.jpg')
input_file = 'input.jpg'
output_file = 'blurred_output.jpg'

# 이미지 불러오기
image = cv2.imread(input_file)

# 가우시안 블러 적용 (커널 크기 (15, 15), 표준편차 0)
blurred = cv2.GaussianBlur(image, (41, 41), 7)

# 결과 이미지 저장
cv2.imwrite(output_file, blurred)

print(f"가우시안 블러가 적용된 이미지를 '{output_file}'로 저장했습니다.")
