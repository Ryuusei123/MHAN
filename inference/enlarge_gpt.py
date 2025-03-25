import cv2
import os


def draw_local_zoom_callback(value):
    global zoom_fac
    zoom_fac = value
    print("Zoom fac: {}".format(zoom_fac))


def mouse_callback(event, x1, y1, flags, userdata):
    global pt_l
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Rec left top: {} {}".format(x1, y1))
        pt_l.append([x1, y1])


def draw_zoom_rec(img, save_zoom=False, image_name=None, output_path=None):
    global zoom_fac
    h, w = img.shape[0], img.shape[1]
    x1, y1, x2, y2 = pt_l[0][0], pt_l[0][1], pt_l[1][0], pt_l[1][1]
    zoom_w = x2 - x1
    zoom_h = y2 - y1

    # 放大系数弹窗自定义
    if zoom_w * zoom_fac > w or zoom_h * zoom_fac > h:
        print("Local zoom size > image size, Please set smaller zoom size")
        return
    zoom_w = int(zoom_w * zoom_fac)
    zoom_h = int(zoom_h * zoom_fac)

    # 提取选中区域并放大
    zoom_in_img = img[y1:y2, x1:x2]
    zoom_in_img = cv2.resize(zoom_in_img, (zoom_w, zoom_h))

    # 单独保存放大的局部图像
    if save_zoom and image_name and output_path:
        zoom_file_name = os.path.join(output_path, image_name.split('.')[0] + "_zoom.png")
        cv2.imwrite(zoom_file_name, zoom_in_img)
        print(f"Zoomed image saved to {zoom_file_name}")

    # 在原图上绘制矩形和放大区域
    cv2.rectangle(img, pt_l[0], pt_l[1], (0, 255, 0), 3, -1)



if __name__ == '__main__':
    ws_path = os.path.abspath(".")
    data_path = os.path.join(ws_path, "imgs/Urban_083")
    if not os.path.exists(data_path):
        raise Exception("Data path not exist!")
    img_file_list = os.listdir(data_path)
    if not img_file_list:
        raise Exception("Data folder has no image files.")
    cv2.namedWindow("DrawRec", flags=0)
    cv2.setMouseCallback("DrawRec", mouse_callback)
    cv2.createTrackbar("ZoomSize", "DrawRec", 3, 5, draw_local_zoom_callback)
    pt_l = []
    zoom_fac = 3
    output_path = os.path.join(ws_path, "output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    while True:
        vis_img = cv2.imread(os.path.join(data_path, img_file_list[0]))
        cv2.imshow("DrawRec", vis_img)
        k = cv2.waitKey(1)
        if k == 27:
            break

    for image_file in img_file_list:
        img = cv2.imread(os.path.join(data_path, image_file))
        if len(pt_l) == 2:
            draw_zoom_rec(img, save_zoom=True, image_name=image_file, output_path=output_path)
            cv2.imwrite(os.path.join(output_path, image_file.split('.')[0] + "rec.png"), img)
    pt_l.clear()
    cv2.destroyAllWindows()
