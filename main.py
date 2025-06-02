import cv2
import numpy as np
import os

def find_top_face_contour(image, debug=False):
    """
    یافتن کانتور وجه بالای تاس (بزرگ‌ترین چهارضلعی با نسبت ابعاد مناسب).
    """
    # پیش‌پردازش تصویر
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # تشخیص لبه‌ها با تنظیم پارامترهای Canny
    edges = cv2.Canny(blurred, 20, 80, apertureSize=3)
    
    # استفاده از عملیات مورفولوژی برای بهبود لبه‌ها
    kernel = np.ones((5, 5), np.uint8)  # افزایش اندازه کرنل برای اتصال بهتر
    edges = cv2.dilate(edges, kernel, iterations=2)  # افزایش تعداد تکرار dilation
    edges = cv2.erode(edges, kernel, iterations=1)  # اضافه کردن erosion برای کاهش نویز
    
    # یافتن کانتورها
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quadrilaterals = []
    
    if debug:
        print(f"تعداد کانتورهای اولیه: {len(contours)}")
    
    for contour in contours:
        # تقریب به یک چندضلعی
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # بررسی اینکه آیا کانتور چهارضلعی است
        if len(approx) == 4:
            # محاسبه مساحت و نسبت ابعاد
            area = cv2.contourArea(approx)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            if debug:
                print(f"چهارضلعی یافت شد - مساحت: {area}, نسبت ابعاد: {aspect_ratio}")
            
            # شرط‌های آزادتر برای انتخاب چهارضلعی
            if 0.3 < aspect_ratio < 3.0 and area > 200:  # کاهش حداقل مساحت و بازتر کردن نسبت ابعاد
                quadrilaterals.append((approx, area))
            else:
                if debug:
                    print(f"چهارضلعی رد شد - مساحت: {area}, نسبت ابعاد: {aspect_ratio}")
    
    if quadrilaterals:
        # انتخاب بزرگ‌ترین چهارضلعی بر اساس مساحت
        top_face = max(quadrilaterals, key=lambda x: x[1])[0]
        if debug:
            print("وجه بالای تاس با موفقیت تشخیص داده شد.")
        return top_face
    
    if debug:
        print("هیچ چهارضلعی مناسبی یافت نشد.")
        # ذخیره تصویر لبه‌ها برای دیباگ
        cv2.imwrite("debug_edges.png", edges)
    return None

def detect_dice_spots(image_path, output_dir="output", debug=False):
    """
    تشخیص نقاط روی وجه بالای تاس و ذخیره تصویر پردازش‌شده.
    """
    # ایجاد پوشه خروجی اگر وجود نداشته باشد
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # خواندن تصویر
    image = cv2.imread(image_path)
    if image is None:
        print(f"خطا: تصویر {image_path} یافت نشد!")
        return None, None
    
    # کپی از تصویر برای رسم
    image_copy = image.copy()
    
    # یافتن وجه بالای تاس
    top_face = find_top_face_contour(image, debug=debug)
    if top_face is None:
        print(f"خطا: نتوانستیم وجه بالای تاس را در {image_path} تشخیص دهیم.")
        return None, None
    
    # اطمینان از قالب صحیح کانتور
    if not isinstance(top_face, np.ndarray):
        top_face = np.array(top_face, dtype=np.int32)
    if top_face.shape[-1] != 2 or len(top_face.shape) != 3:
        top_face = top_face.reshape(-1, 1, 2)
    
    # پیش‌پردازش تصویر برای تشخیص نقاط
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)  # استفاده از Median Blur برای کاهش نویز و بازتاب نور
    
    # تنظیم پارامترهای SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    
    # فیلتر بر اساس مساحت
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 1000
    
    # فیلتر بر اساس دایره‌ای بودن
    params.filterByCircularity = True
    params.minCircularity = 0.7
    
    # فیلتر بر اساس محدب بودن
    params.filterByConvexity = True
    params.minConvexity = 0.87
    
    # فیلتر بر اساس اینرسی
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    # ایجاد دتکتور
    detector = cv2.SimpleBlobDetector_create(params)
    
    # تشخیص نقاط
    keypoints = detector.detect(blurred)
    
    if debug:
        print(f"تعداد نقاط تشخیص‌داده‌شده: {len(keypoints)}")
    
    # شمارش نقاط داخل وجه بالای تاس
    dice_number = 0
    for keypoint in keypoints:
        x, y = keypoint.pt
        point = (float(x), float(y))
        
        # بررسی اینکه آیا نقطه داخل کانتور است
        try:
            if cv2.pointPolygonTest(top_face, point, False) >= 0:  # داخل یا روی کانتور
                dice_number += 1
                # رسم دایره سبز دور نقاط
                cv2.circle(image_copy, (int(x), int(y)), int(keypoint.size / 2), (0, 255, 0), 2)
                # رسم نقطه مرکزی قرمز
                cv2.circle(image_copy, (int(x), int(y)), 2, (0, 0, 255), 3)
        except cv2.error as e:
            print(f"خطا در pointPolygonTest برای نقطه ({x}, {y}): {e}")
            continue
    
    # رسم کانتور وجه بالای تاس
    cv2.drawContours(image_copy, [top_face], -1, (255, 0, 0), 2)  # رسم کانتور آبی
    
    # ذخیره تصویر پردازش‌شده
    output_filename = f"processed_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, image_copy)
    
    print(f"تصویر: {image_path}")
    print(f"تعداد نقاط روی وجه بالای تاس: {dice_number}")
    print(f"تصویر پردازش‌شده ذخیره شد در: {output_path}\n")
    
    return dice_number, image_copy

def main():
    # لیست تصاویر ورودی
    image_paths = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png']
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            # فعال کردن دیباگ برای بررسی دقیق‌تر
            detect_dice_spots(image_path, debug=True)
        else:
            print(f"خطا: فایل {image_path} وجود ندارد!\n")

if __name__ == "__main__":
    main()