import cv2
import numpy as np
import os
import time

# 灰度化
def Grayscale(original_img):
    img = original_img.copy()
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray_img

# 二值化
def Binarization(image):
    Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    img = cv2.convertScaleAbs(Sobel_x)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    return img

# 截取车牌
def my_Locate(original_img):
    original_img = cv2.resize(original_img, (450, 600))
    r, c = np.shape(original_img)[0], np.shape(original_img)[1]

    # blue_num_r = [0 for idx in range(r)]
    # blue_num_c = [0 for idx in range(c)]
    # for i in range(r):
    #     for j in range(c):
    #         if original_img[i, j, 0] > 256//3 \
    #                 and original_img[i, j, 1] / (original_img[i, j, 0] + 0.5) < 0.7 \
    #                 and original_img[i, j, 2] / (original_img[i, j, 0] + 0.5) < 0.7:
    #             blue_num_r[i] += 1
    #             blue_num_c[j] += 1

    flag1 = original_img[:, :, 0] > 256 // 3
    flag2 = original_img[:, :, 1] / (original_img[:, :, 0] + 0.5) < 0.7
    flag3 = original_img[:, :, 2] / (original_img[:, :, 0] + 0.5) < 0.7
    flag = flag1 & flag2 & flag3
    blue_num_c = np.sum(flag, axis=0)
    blue_num_r = np.sum(flag, axis=-1)

    br_max = max(blue_num_r)
    bc_max = max(blue_num_c)

    posr = np.where(blue_num_r > br_max // 2)
    r1, r2 = posr[0][0], posr[0][-1]

    posc = np.where(blue_num_c > bc_max // 4)
    c1, c2 = posc[0][0], posc[0][-1]

    # for i, br in enumerate(blue_num_r):
    #     if br > br_max//2:
    #         r1 = i
    #         break
    # i = len(blue_num_r)
    # while i > r1:
    #     if blue_num_r[i-1] > br_max//2:
    #         r2 = i-1
    #         break
    #     i -= 1
    #
    # for j, bc in enumerate(blue_num_c):
    #     if bc > bc_max//4:
    #         c1 = j
    #         break
    # j = len(blue_num_c)
    # while j > c1:
    #     if blue_num_c[j-1] > bc_max//4:
    #         c2 = j-1
    #         break
    #     j -= 1

    return original_img[r1:r2, c1:c2]

# 分割字符
def my_Segmentation(image, ord):
    img = cv2.GaussianBlur(image, (3, 3), 0)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    img = cv2.resize(img, (140, 50))
    # plt.imshow(img, cmap='gray')
    # plt.show()

    cols = np.shape(img)[1]
    word_images = []
    for i in range(7):
        split_img = cv2.resize(img[:, i*(cols//7):(i+1)*(cols//7)], (20, 20))
        word_images.append(split_img)


    return word_images

def get_Template_Filename(template_directoryname):
    template_filename = []
    for f in os.listdir(template_directoryname):
        template_filename.append(template_directoryname + "/" + f)

    return template_filename

def get_Chinese_Words(template):
    Chinese_Words = []
    for i in range(34, 65):
        Chinese_Words.append(get_Template_Filename("./static/imgs/Template/" + template[i]))

    return Chinese_Words

def get_Cities(template):
    Cities = []
    for i in range(10, 34):
        Cities.append(get_Template_Filename("./static/imgs/Template/" + template[i]))

    return Cities

def get_Letters_Numbers(template):
    Letters_Numbers = []
    for i in range(34):
        Letters_Numbers.append(get_Template_Filename("./static/imgs/Template/" + template[i]))

    return Letters_Numbers

def get_Score(template_image, image):
    template_img = cv2.imdecode(np.fromfile(template_image, dtype=np.uint8), 1)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)

    # img = image.copy()
    result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)

    return result[0][0]

def Match_Template(word_images, template):
    result = []
    for i, word_image in enumerate(word_images):
        if i == 0:
            Score = []
            CWS = get_Chinese_Words(template)
            for CW in CWS:
                score = []
                for cw in CW:
                    score.append(get_Score(cw, word_image))
                Score.append(max(score))
            j = Score.index(max(Score))
            result.append(template[34+j])

        elif i == 1:
            Score = []
            Cities = get_Cities(template)
            for Citie in Cities:
                score = []
                for city in Citie:
                    score.append(get_Score(city, word_image))
                Score.append(max(score))
            j = Score.index(max(Score))
            result.append(template[10 + j])

        else:
            Score = []
            LNS = get_Letters_Numbers(template)
            for LN in LNS:
                score = []
                for ln in LN:
                    score.append(get_Score(ln, word_image))
                Score.append(max(score))
            j = Score.index(max(Score))
            result.append(template[j])

    return result

def main():

    labels = open('./test/test_label.txt', "r")
    cnt = 0
    time1 = time.time()

    for ord in range(1, 101):

        t1 = time.time()
        src_path = "./test/测试车牌" + str(ord) + ".jpg"
        template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                    'X', 'Y', 'Z',
                    '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
                    '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']

        original_img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), -1)
        rows = np.shape(original_img)[0]
        cv2.imwrite("./test/" + str(ord) + ".png", original_img[rows//5:rows*5//6, :])
        original_img = cv2.imdecode(np.fromfile("./test/" + str(ord) + ".png", dtype=np.uint8), -1)
        original_img = my_Locate(original_img)

        imgs = my_Segmentation(original_img, ord)
        judge_res = Match_Template(imgs, template)
        t2 = time.time()

        label = labels.readline()[:-1]
        flag = label == "".join(judge_res)
        if flag:
            flag = "OK"
            cnt+=1
        else:
            flag = "False"
        print(ord, "".join(judge_res), flag)
    print("准确率：", cnt / 100 * 100, "%.")
    print("总耗时：", time.time() - time1, "sec.")


def reg(path):
    t1 = time.time()
    src_path = path
    template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                'X', 'Y', 'Z',
                '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
                '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']

    original_img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), -1)
    original_img = cv2.imread(src_path)
    rows = np.shape(original_img)[0]
    cv2.imwrite("./test/" + str(t1) + ".png", original_img[rows // 5:rows * 5 // 6, :])
    original_img = cv2.imdecode(np.fromfile("./test/" + str(t1) + ".png", dtype=np.uint8), -1)
    original_img = original_img[rows//5:rows*5//6, :]
    original_img = my_Locate(original_img)

    imgs = my_Segmentation(original_img, ord)
    judge_res = Match_Template(imgs, template)
    t2 = time.time()
    print(ord, "".join(judge_res), t2 - t1)
    return judge_res

if __name__ == '__main__':
    main()
