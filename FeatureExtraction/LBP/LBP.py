import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Img:
    def __init__(self, img_matrix, radius=1, sampling_points=8):
        self.img_matrix = img_matrix

        self.radius = radius
        self.sampling_points = sampling_points

        self.hash_uniform_patterns, self.uniform_patterns = ImgProcess.get_patterns(sampling_points)
        self.lbp_basic = None
        self.lbp_min_ror = None
        self.lbp_uniform = None
        self.texture_img_matrix = None
        self.lbp_matrix = None
        self.local_imgs = None
        self.global_histogram = None
        self.local_histogram = None
        self.frequency = None
        self.local_frequency = None


class ImgProcess:
    @staticmethod
    def get_global_frequency(img: Img):
        if img.lbp_basic is not None:
            row, col = img.lbp_basic.shape
            img.frequency = [0] * int((math.pow(2, 8)))
            for i in range(row):
                for j in range(col):
                    img.frequency[int(img.lbp_basic[i, j])] += 1
        else:
            row, col = img.lbp_uniform.shape
            img.frequency = [0] * (img.sampling_points + 2)
            for i in range(row):
                for j in range(col):
                    img.frequency[int(img.lbp_uniform[i, j])] += 1
                    print(img.lbp_uniform[i, j])

    @staticmethod
    def get_local_frequency(img: Img):
        img.local_frequency = list()
        for local_img in img.local_imgs:
            ImgProcess.get_global_frequency(local_img)
            img.local_frequency.append(local_img.frequency)

    @staticmethod
    def img_preprocessing(img_path):
        img_pil = Image.open(img_path).convert('L')
        return np.matrix(img_pil)

    @staticmethod
    def lbp_basic(img: Img):
        img_matrix = img.img_matrix
        row, col = img_matrix.shape
        result = np.zeros((row - 2, col - 2))
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                point = img_matrix[i, j]
                value = 0
                for count in range(9):
                    if count != 4:
                        num = 1 if img_matrix[i - 1 + count // 3, j - 1 + count % 3] > point else 0
                        value = (value << 1) | num
                result[i - 1, j - 1] = np.int8(ImgProcess.get_min(value, 8))
                # result[i - 1, j - 1] = np.int8(value)
        img.texture_img_matrix = result
        img.lbp_basic = result

    @staticmethod
    def lbp_improved(img: Img):
        img_matrix = img.img_matrix
        radius = img.radius
        sampling_points = img.sampling_points
        scale = math.pow(2, 8) / math.pow(2, sampling_points)
        row, col = img_matrix.shape
        print(row, col, row - radius, col - radius)

        lbp_min_ror = np.zeros((row - 2 * radius, col - 2 * radius))
        lbp_uniform = np.zeros((row - 2 * radius, col - 2 * radius))
        texture = np.zeros((row - 2 * radius, col - 2 * radius))
        for i in range(radius, row - radius):

            for j in range(radius, col - radius):
                print(i, j)
                point = img_matrix[i, j]
                value = 0
                for p in range(0, sampling_points):
                    x_p = i + radius * math.sin(2 * math.pi * p / sampling_points)
                    y_p = j - radius * math.cos(2 * math.pi * p / sampling_points)
                    if (x_p - int(x_p)) == 0 and (y_p - int(y_p)) == 0:
                        value_p = img_matrix[int(x_p), int(y_p)]
                    else:
                        x_0 = math.floor(x_p)
                        y_0 = math.floor(y_p)
                        x_1 = math.floor(x_p + 1 * math.copysign(1, (i - x_p)))
                        y_1 = math.floor(y_p + 1 * math.copysign(1, (j - y_p)))
                        value_p = abs((np.matrix([y_p - y_0, y_1 - y_p]) @
                                       np.matrix([[img_matrix[x_0, y_0], img_matrix[x_1, y_0]],
                                                  [img_matrix[x_0, y_1], img_matrix[x_1, y_1]]]) @
                                       np.matrix([[x_p - x_0], [x_1 - x_p]]))[0, 0])
                    num = 1 if value_p > point else 0
                    value = (value << 1) | num
                texture[i - radius, j - radius] = np.int8(ImgProcess.get_min(value, sampling_points) * scale)
                lbp_min_ror[i - radius, j - radius] = ImgProcess.get_min(value, sampling_points)
                lbp_uniform[i - radius, j - radius] = img.hash_uniform_patterns[str(int(value))]
        img.lbp_min_ror = lbp_min_ror
        img.lbp_uniform = lbp_uniform
        img.texture_img_matrix = texture

    @staticmethod
    def split_img(img, splitting_size):
        row, col = img.img_matrix.shape
        row_boundary = int((row // splitting_size) * splitting_size)
        col_boundary = int((col // splitting_size) * splitting_size)
        imgs_split = list()
        i = 0
        j = 0
        while i < row_boundary:
            while j < col_boundary:
                local_img = Img(img.img_matrix[i:i + splitting_size, j:j + splitting_size], img.radius,
                                img.sampling_points)
                imgs_split.append(local_img)
                j += splitting_size
            i += splitting_size
        img.local_imgs = imgs_split

    @staticmethod
    def get_min(num, sampling_points):
        num = bin(num)
        binary_string = '0' * (sampling_points - len(str(num)[2:])) + str(num)[2:]
        num_list = list()
        for i in range(sampling_points):
            num_list.append(int(binary_string[i:] + binary_string[0:i], 2))
        return min(num_list)

    @staticmethod
    def get_num_of_shift(binary_string):
        num_of_shift = 0
        for i in range(len(binary_string)):
            num_of_shift += abs(int(int(binary_string[i]) - int(binary_string[i - 1])))
        return num_of_shift

    @staticmethod
    def get_patterns(sample_points):
        hash_patterns = dict()
        used = set()
        for num in range(int(math.pow(2, sample_points))):
            if num in used:
                continue

            binary_num = bin(num)
            binary_string = '0' * (sample_points - len(str(binary_num[2:]))) + str(binary_num)[2:]
            if ImgProcess.get_num_of_shift(binary_string) <= 2:
                bit = int(ImgProcess.get_num_of_one_bit(binary_string))
                for i in range(sample_points):
                    tmp = int(binary_string[i:] + binary_string[0:i], 2)
                    hash_patterns[str(tmp)] = bit
                    used.add(tmp)

            else:
                for i in range(sample_points):
                    tmp = int(binary_string[i:] + binary_string[0:i], 2)
                    hash_patterns[str(tmp)] = sample_points + 1
                    used.add(tmp)

        return hash_patterns, list(range(0, sample_points + 2))

    @staticmethod
    def get_num_of_one_bit(binary_string):
        num = 0
        for x in binary_string:
            num += int(x)
        return num

    @staticmethod
    def img_show(img_matrix):
        plt.figure()
        plt.axis('off')
        plt.imshow(img_matrix, cmap='gray')


def mytest_improve(img: Img):
    ImgProcess.lbp_improved(img)
    ImgProcess.img_show(img.texture_img_matrix)
    ImgProcess.get_global_frequency(img)
    # plt.savefig('./pic/result/' + str(img.radius) + '_' + str(img.sampling_points) + '.png')


def mytest_basic(img):
    ImgProcess.lbp_basic(img)
    # ImgProcess.img_show(img.texture_img_matrix)
    ImgProcess.get_global_frequency(img)
    # plt.savefig('./pic/result/BASIC_ROR' + str(img.radius) + '_' + str(img.sampling_points) + '.png')


if __name__ == '__main__':
    # print(ImgProcess.get_patterns(8))
    t_matrix = ImgProcess.img_preprocessing("./pic/test/3.jpg")
    test1 = Img(t_matrix, radius=1, sampling_points=8)
    mytest_improve(test1)
    # test2 = Img(t_matrix, radius=2, sampling_points=16)
    # test3 = Img(t_matrix, radius=3, sampling_points=24)
    # test1_process = multiprocessing.Process(target=test_improve, args=(test1, ))
    # test2_process = multiprocessing.Process(target=test_improve, args=(test2, ))
    # test3_process = multiprocessing.Process(target=test_improve, args=(test3, ))
    #
    # test1_process.start()
    # test2_process.start()
    # test3_process.start()
    print(test1.frequency)
    plt.bar(x=test1.uniform_patterns, height=list(map(lambda x: x/sum(test1.frequency), test1.frequency)), width=0.3)
    plt.show()
