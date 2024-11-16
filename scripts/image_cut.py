import cv2

# 读取图片
img1 = cv2.imread('/home/saev/changye/blue.jpg')
img2 = cv2.imread('/home/saev/changye/yellow.png')

# 获取图片的高度和宽度
height1, width1, _ = img1.shape
height2, width2, _ = img2.shape

height = min(height1, height2)  # 选择最小的高度
width=min(width1,width2)
img1_left_resized = cv2.resize(img1, (width, height))
img2_right_resized = cv2.resize(img2, (width, height))
# # 计算裁剪的中间位置
# half_width = width // 2

# # 裁剪每张图片的左半部分
# img1_left = img1_left_resized [:, :half_width]
# img2_right = img2_right_resized[:, half_width:]

# # 水平拼接裁剪后的部分
# result = cv2.hconcat([ img2_right,img1_left])
half_height = height // 2

# 上半部分裁剪
img1_top = img1_left_resized[:half_height, :]
img2_bottom = img2_right_resized[half_height:, :]

# 垂直拼接裁剪后的部分
result = cv2.vconcat([ img2_bottom,img1_top])

# 保存结果
cv2.imwrite('yellow_top.jpg', result)
