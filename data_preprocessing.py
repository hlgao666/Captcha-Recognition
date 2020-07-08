from PIL import Image
import requests
import io
import os


# 数据爬取
def data_spider(gif_save_path, gif_num, img_path):

    url = "https://pass.hust.edu.cn/cas/code"
    # 创建文件夹
    if not os.path.exists(gif_save_path):
        os.makedirs(gif_save_path)

    num = 0
    while num < gif_num:

        # 保存动图
        response = requests.get(url)
        res = response.content
        with open(f'{gif_save_path}/{num}.gif', 'wb') as file:
            file.write(res)

        # 保存图片
        # 创建新的文件夹
        folder_name = img_path + f'数据{num}'
        os.makedirs(folder_name)
        img = Image.open(io.BytesIO(res))

        # GIF --> PNG
        i = 0
        try:
            while i < 4:
                save_place = folder_name + '/' + f'{i}.png'

                # img = img.convert('L')
                img.save(save_place)
                i += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        num += 1


# 灰度处理 & 二值化
def binary_image(filedir):
    for root, dirs, files in os.walk(filedir):

        # 遍历所有文件
        for d in dirs:
            for num in range(0, 4):
                img_path = os.path.join(root, d) + '/' + f'{num}.png'
                img_raw = Image.open(img_path)
                img = img_raw.convert('L')  # 灰度处理L模式
                # 二值化
                table = get_binary_table()
                img = img.point(table, '1')
                img.save(img_path)


# 二值化
def get_binary_table(threshold=200):
    """
    获取灰度转二值的映射table
    0表示黑色,1表示白色
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    return table


# 分割图片字符
def get_seg_imgs(img):

    child_img_list = []
    for i in range(4):
        x = i * (16 + 5)
        y = 19
        child_img = img.crop((x, y, x + 16, y + 20))
        child_img_list.append(child_img)

    return child_img_list


# 九宫格法去噪
class ClearNoise:

    def __init__(self, filedir):
        self.filedir = filedir

    def sum_9_region_new(img, x, y):
        # 确定噪点
        cur_pixel = img.getpixel((x, y))  # 当前像素点的值
        width = img.width
        height = img.height

        if cur_pixel == 1:  # 如果当前点为白色区域,则不统计邻域值
            return 0

        # 因当前图片的四周都有黑点，所以周围的黑点可以去除
        if y < 19:  # 本例中，前后19行的黑点都可以去除
            return 1
        elif y > height - 19:  # 最下面19行
            return 1
        else:  # y不在边界
            if x == 0:  # 左边非顶点列
                sum = img.getpixel((x, y - 1)) \
                      + cur_pixel \
                      + img.getpixel((x, y + 1)) \
                      + img.getpixel((x + 1, y - 1)) \
                      + img.getpixel((x + 1, y)) \
                      + img.getpixel((x + 1, y + 1))
                return 6 - sum
            elif x >= width - 8:  # 右边非顶点, 最右8列无内容
                return 1
            else:  # 具备9领域条件的顶点
                sum = img.getpixel((x - 1, y - 1)) \
                      + img.getpixel((x - 1, y)) \
                      + img.getpixel((x - 1, y + 1)) \
                      + img.getpixel((x, y - 1)) \
                      + cur_pixel \
                      + img.getpixel((x, y + 1)) \
                      + img.getpixel((x + 1, y - 1)) \
                      + img.getpixel((x + 1, y)) \
                      + img.getpixel((x + 1, y + 1))
                return 9 - sum

    def collect_noise_point(img):
        # 收集所有的噪点
        noise_point_list = []
        for x in range(img.width):
            for y in range(img.height):
                res_9 = ClearNoise.sum_9_region_new(img, x, y)
                if (0 <= res_9 < 4) and img.getpixel((x, y)) == 0:  # 找到孤立点
                    pos = (x, y)
                    noise_point_list.append(pos)
        return noise_point_list

    def remove_noise_pixel(img, noise_point_list):
        # 根据噪点的位置信息，消除二值图片的黑点噪声
        for item in noise_point_list:
            img.putpixel((item[0], item[1]), 1)

    def clear_noise(self, seg_imgs_path, mode):

        for root, dirs, files in os.walk(self.filedir):
            # 创建'D:/dataSet'文件夹

            if not os.path.exists(seg_imgs_path):
                os.makedirs(seg_imgs_path)

            count = 0
            # 遍历所有文件
            for d in dirs:
                for num in range(0, 4):
                    img_path = os.path.join(root, d) + '/' + f'{num}.png'
                    img_raw = Image.open(img_path)
                    # 去噪
                    noise_point_list = ClearNoise.collect_noise_point(img_raw)
                    ClearNoise.remove_noise_pixel(img_raw, noise_point_list)
                    img_raw.save(img_path)
                    # print(f'{img_path} have finished')
                    # 分割，过滤
                    child_img_list = get_seg_imgs(img_raw)
                    for img in child_img_list:
                        # 训练模式
                        if mode == 0:
                            # 过滤无效图片（黑色像素点小于80为无效）
                            black_count = 0
                            for x in range(img.width):
                                for y in range(img.height):
                                    if img.getpixel((x, y)) == 0:
                                        black_count += 1
                            # 黑色像素点数大于80才保存
                            if black_count > 80:
                                img.save(f'{seg_imgs_path}/{count}.png')
                                count += 1
                        # 测试模式
                        else:
                            img.save(f'{seg_imgs_path}/{count}.png')
                            count += 1
