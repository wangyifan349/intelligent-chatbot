import cv2
import numpy as np
import dlib

# 人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 获取人脸关键点
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    faces = detector(gray)  # 检测人脸
    if len(faces) == 0:  # 没有检测到人脸
        return None
    shape = predictor(gray, faces[0])  # 获取第一个人脸的关键点
    points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]  # 提取68个关键点
    return np.array(points)

# 高斯模糊 + 双边滤波实现磨皮
def enhanced_smoothing(image, landmarks, gaussian_kernel=(5, 5), bilateral_d=15, bilateral_sigmaColor=80, bilateral_sigmaSpace=80):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 创建脸部掩膜
    hull = cv2.convexHull(landmarks[:17])  # 提取脸部凸包轮廓
    cv2.fillPoly(mask, [hull], 255)  # 填充脸部区域
    face_only = cv2.bitwise_and(image, image, mask=mask)  # 提取脸部区域

    smooth_face = cv2.bilateralFilter(face_only, d=bilateral_d, sigmaColor=bilateral_sigmaColor, sigmaSpace=bilateral_sigmaSpace)  # 双边滤波
    gaussian_blurred_face = cv2.GaussianBlur(smooth_face, gaussian_kernel, 0)  # 高斯模糊

    result = np.where(mask[:, :, None] == 255, gaussian_blurred_face, image)  # 在脸部区域应用平滑
    return result

# 皮肤亮度增强（美白）
def brighten_skin(image, landmarks, brightness=25):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 创建脸部掩膜
    hull = cv2.convexHull(landmarks[:17])  # 提取脸部凸包轮廓
    cv2.fillPoly(mask, [hull], 255)  # 填充脸部区域

    img_float = image.astype(np.float32)  # 转为浮点数便于计算
    img_float[mask == 255] = np.clip(img_float[mask == 255] + brightness, 0, 255)  # 提高亮度
    img = img_float.astype(np.uint8)  # 转回无符号8位整型

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)  # 转为HSV空间
    hsv[..., 1] *= 0.85  # 降低饱和度
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)  # 限定饱和度范围
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)  # 转回BGR空间

    img[mask != 255] = image[mask != 255]  # 非脸部区域保持原图
    return img

# 去除小斑点或瑕疵
def remove_spots(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 创建脸部掩膜
    hull = cv2.convexHull(landmarks[:17])  # 提取脸部凸包轮廓
    cv2.fillPoly(mask, [hull], 255)  # 填充脸部区域

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # 转为LAB色彩空间
    l = lab[:, :, 0]  # 亮度通道
    face_l = cv2.bitwise_and(l, l, mask=mask)  # 提取脸部亮度
    blur = cv2.GaussianBlur(face_l, (5, 5), 0)  # 高斯模糊
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # 自适应阈值
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)  # 只保留脸部区域

    kernel = np.ones((3, 3), np.uint8)  # 形态学核
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算去除噪点

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    inpaint_mask = np.zeros_like(mask)  # 创建修复掩膜
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 200:  # 仅处理面积在指定范围内的瑕疵
            cv2.drawContours(inpaint_mask, [cnt], -1, 255, -1)  # 标记修复区域

    if np.any(inpaint_mask):  # 如果存在需要修复的区域
        return cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)  # 使用Inpainting修复
    return image

# 瘦脸变形处理
def slim_face(image, landmarks, factor=0.15):
    points = landmarks.astype(np.float32)  # 转为浮点型
    img_h, img_w = image.shape[:2]
    dst_points = points.copy()

    center_line_x = (points[3][0] + points[13][0]) / 2  # 面部中心线X坐标
    for i in range(0, 4):  # 左脸关键点
        dst_points[i][0] += factor * (center_line_x - points[i][0])
    for i in range(13, 17):  # 右脸关键点
        dst_points[i][0] += factor * (center_line_x - points[i][0])

    rect = (0, 0, img_w, img_h)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))

    triangles = subdiv.getTriangleList()
    delaunay_tri = []
    for tri in triangles:
        pts = [(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])]
        idx = []
        for p in pts:
            for j, point in enumerate(points):
                if abs(p[0] - point[0]) < 1 and abs(p[1] - point[1]) < 1:
                    idx.append(j)
        if len(idx) == 3:
            delaunay_tri.append(idx)

    result = np.zeros_like(image)
    for tri in delaunay_tri:
        x, y, z = tri
        src_tri = np.float32([points[x], points[y], points[z]])
        dst_tri = np.float32([dst_points[x], dst_points[y], dst_points[z]])

        r1 = cv2.boundingRect(src_tri)
        r2 = cv2.boundingRect(dst_tri)

        src_rect = src_tri - r1[:2]
        dst_rect = dst_tri - r2[:2]

        img_crop = image[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        warp_mat = cv2.getAffineTransform(src_rect, dst_rect)
        img_warp = cv2.warpAffine(img_crop, warp_mat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_rect), (1.0, 1.0, 1.0))

        result[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img_warp * mask + result[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask)

    return result

# 眼睛提亮
def brighten_eyes(image, landmarks, bright=40):
    result = image.copy()
    for eye_pts in [landmarks[36:42], landmarks[42:48]]:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(eye_pts, dtype=np.int32)], 255)

        x, y, w, h = cv2.boundingRect(np.array(eye_pts, dtype=np.int32))
        roi = result[y:y + h, x:x + w]
        
        eye_mask = mask[y:y + h, x:x + w] / 255.0
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + bright * eye_mask, 0, 255)
        
        result[y:y + h, x:x + w] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result

# 综合美颜处理
def apply_beauty(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return

    landmarks = get_landmarks(img)
    if landmarks is None:
        print("No face detected")
        cv2.imwrite(output_path, img)
        return

    res = enhanced_smoothing(img, landmarks, gaussian_kernel=(7, 7), bilateral_d=25, bilateral_sigmaColor=100, bilateral_sigmaSpace=100)
    res = brighten_skin(res, landmarks, brightness=30)
    res = remove_spots(res, landmarks)
    res = slim_face(res, landmarks, factor=0.15)
    res = brighten_eyes(res, landmarks, bright=40)

    cv2.imwrite(output_path, res)
    comp = np.hstack((img, res))
    cv2.imshow("Before | After", comp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    apply_beauty("input.jpg", "output.jpg")
