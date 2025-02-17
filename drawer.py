import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import io
from PIL import Image
from imageio.v2 import imread, imwrite
from time import sleep, time

from skimage.measure import label, regionprops
from skimage.feature import canny
from skimage.transform import resize, rotate
from skimage.morphology import binary_dilation, square

import ctypes as c

from plotter import Plotter

from multiprocessing import Process, Queue

import cv2
from math import sin, cos, acos, sqrt, pi
from detect_paper import detect_paper

from serial import Serial

class TrajectoryLib:
    def __init__(
        self, 
        cpp_path, 
        lib_folder, 
        cuda_version = '12.5', 
        gpu_arch = '86'
    ):
        self.cpp_path = cpp_path
        self.lib_folder = lib_folder
        self.cuda_version = cuda_version
        self.gpu_arch = gpu_arch
        
        self.compile_lib()
        self.load_lib()
        
    def compile_lib(self):
        # Открываем исходный файл и файл предыдущей версии
        src = open(self.cpp_path, 'r')
        dst = open(f'{self.lib_folder}/trajectory.prev', 'r')
        s, d = src.read(), dst.read()

        try:
            # Сравниваем содержимое файлов
            if s != d:
                print('compiling...')
                dst.close()
                
                # Компилируем исходный файл в объектный файл и библиотеку
                err1 = os.system(
                    'nvc++ -fPIC -stdpar -Iinclude-stdpar '
                    f'-gpu=managed,cuda{self.cuda_version},cc{self.gpu_arch} ' 
                    f'-std=c++17 -c {self.cpp_path} -o {self.lib_folder}/trajectory.o'
                )
                err2 = os.system(
                    'nvc++ -shared '
                    f'-gpu=managed,cuda{self.cuda_version},cc{self.gpu_arch} '
                    f'-stdpar {self.lib_folder}/trajectory.o -o {self.lib_folder}/trajectory.so')
                
                # Если произошла ошибка компиляции, завершаем программу с кодом ошибки
                if err1: exit(err1)
                if err2: exit(err2)
                
                # Сохраняем текущую версию файла как предыдущую
                dst = open(f'{self.lib_folder}/trajectory.prev', 'w')
                dst.write(s)
                print('compiled')
        finally:
            # Закрываем файлы
            dst.close()
            src.close()
            
    def load_lib(self):
        # Загружаем скомпилированную библиотеку
        self.lib = c.CDLL(f'{self.lib_folder}/trajectory.so')

        # Определяем типы указателей для функций библиотеки
        DPOINTER2D = np.ctypeslib.ndpointer(
            dtype=np.float128,
            ndim=2,
            flags='C'
        )

        DPOINTER3D = np.ctypeslib.ndpointer(
            dtype=np.float128,
            ndim=3,
            flags='C'
        )

        IPOINTER2D = np.ctypeslib.ndpointer(
            dtype=np.int32,
            ndim=2,
            flags='C'
        )
        
        self.lib.mark_image_with_colors.argtypes = [DPOINTER2D, DPOINTER2D, c.c_size_t, c.c_size_t, c.c_size_t, c.c_size_t]
        self.lib.mark_image_with_colors.restype = None

        self.lib.fill_image_area.argtypes = [c.c_int32, c.c_int32, IPOINTER2D, DPOINTER3D, c.c_size_t, c.c_size_t, c.c_size_t]
        self.lib.fill_image_area.restype = None

        self.lib.compute_image_trajectory.argtypes = [IPOINTER2D, c.c_size_t, c.c_size_t, c.c_int32, c.c_int32, c.c_int32]
        self.lib.compute_image_trajectory.restype = c.POINTER(c.c_int32)
    
    def mark(self, img, clrs):
        """
        Отмечает цвета на изображении.
        
        Параметры:
        img (np.ndarray): Изображение.
        clrs (np.ndarray): Цвета.
        
        Возвращает:
        np.ndarray: Изображение с отмеченными цветами.
        """
        self.lib.mark_image_with_colors(
            img.astype('float128', order='C'), 
            clrs.astype('float64', order='C'), 
            *img.shape, *clrs.shape
        )
        return img
    
    def fill(self, x, y, vis, img):
        """
        Заполняет изображение цветами.
        
        Параметры:
        x (int): Координата x.
        y (int): Координата y.
        vis (np.ndarray): Массив посещённых клеток.
        img (np.ndarray): Изображение.
        
        Возвращает:
        tuple: Визуализация и изображение.
        """
        self.lib.fill_image_area(
            x, y, 
            vis.astype('int32', order='C'), 
            img.astype('float64', order='C'), 
            *img.shape
        )
        return vis, img
    
    def compute_image(self, img, d, sx, sy):
        """
        Вычисляет изображение.
        
        Параметры:
        img (np.ndarray): Изображение.
        d (int): Параметр d.
        sx (int): Координата x начала.
        sy (int): Координата y начала.
        
        Возвращает:
        list: Список координат.
        """
        res = self.lib.compute_image_trajectory(
            img.astype('int32', order='C'), 
            *img.shape, 
            d, sx, sy
        )
        ans = []
        sz = res[0]
        for i in range(sz):
            ans.append([res[i * 2 + 1], res[i * 2 + 2]])

        self.lib.cleanup(res)
    
        return ans
    
def get_colors(img, crop):
    """
    Получает цвета из изображения.
    
    Параметры:
    img (np.ndarray): Изображение.
    crop (bool): Обрезать изображение.
    
    Возвращает:
    np.ndarray: Изображение с цветами.
    """
    if np.max(img) <= 1:
        img *= 255
    t = canny(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY), sigma=2.0)
        
    t = binary_dilation(t, square(5))
    if crop:
        nz = np.nonzero(t)
        t = t[max(0, np.min(nz[0]) - 10):min(t.shape[0], np.max(nz[0]) + 10), max(0, np.min(nz[1]) - 10):min(t.shape[1], np.max(nz[1]) + 10)]
    return t

def shift_coords(sx: float, sy: float, angle: float, matrix: np.ndarray, k: float):
    """
    Сдвигает и масштабирует координаты. 
    
    Параметры:
    sx (float): Координата x начала.
    sy (float): Координата y начала.
    angle (float): Угол поворота.
    matrix (np.ndarray): Матрица координат.
    k (float): Коэффициент масштабирования.
    
    Возвращает:
    np.ndarray: Сдвинутая матрица координат.
    """
    matrix *= k
    rm = np.array([[cos(angle), sin(angle)],
                   [-sin(angle), cos(angle)]])
    rotated = np.matmul(matrix, rm)
    shifted = rotated + [sx, sy]
    return shifted

def get_angle(x1, y1, x2, y2):
    """
    Вычисляет угол между двумя векторами.
    
    Параметры:
    x1 (float): Координата x первого вектора.
    y1 (float): Координата y первого вектора.
    x2 (float): Координата x второго вектора.
    y2 (float): Координата y второго вектора.
    
    Возвращает:
    float: Угол между векторами.
    """
    return acos((x1 * x2 + y1 * y2) / (sqrt(x1 * x1 + y1 * y1) * sqrt(x2 * x2 + y2 * y2))) 

def dist(x1, y1, x2, y2):
    """
    Вычисляет расстояние между двумя точками.
    
    Параметры:
    x1 (float): Координата x первой точки.
    y1 (float): Координата y первой точки.
    x2 (float): Координата x второй точки.
    y2 (float): Координата y второй точки.
    
    Возвращает:
    float: Расстояние между точками.
    """
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    
class Drawer:
    def __init__(
        self,
        stop,
        flag_start,
        images_queue: Queue,
        cpp_path: str, 
        lib_folder: str, 
        video_id: int,
        flag_end,
        cuda_version: str = '12.5', 
        gpu_arch: str = '86',
        port: str = '/dev/ttyACM0',
        baudrate: int = 115200,
    ):
        self.stop = stop
        self.flag_start = flag_start
        self.images_queue = images_queue
        self.cpp_path = cpp_path
        self.lib_folder = lib_folder
        self.video_id = video_id
        self.flag_end = flag_end
        self.cuda_version = cuda_version
        self.gpu_arch = gpu_arch
        self.port = port
        self.baudrate = baudrate
        
    def get_raw_trajectory(self, img):
        idx = 0
        trajectory = []
        lb = label(img)
        rgs = regionprops(lb)
        for reg in rgs:
            idx += 1
            regimg = reg.image
            cords = self.lib.compute_image(regimg, 4, *reg.bbox[:2])
            trajectory.extend(cords)
            if trajectory[-1][0] != 1e9:
                trajectory.append([1e9, 1e9])
        return np.array(trajectory)
    
    def compute_angle(self, show = False):
        points = []
        cnt = 0
        
        self.plotter.move_to(350, 0, 16000)
        
        video = cv2.VideoCapture(self.video_id)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Get ready...")
        try:
            while True:
                ret, frame = video.read()
                
                if not ret:
                    print('Cant')
                    continue
                
                points, frame = detect_paper(frame, warp=True)
                if points is not None:
                    cnt += 1
                    if cnt == 50:
                        break
                if show:
                    cv2.imshow('img', frame)
                    cv2.waitKey(1)
        except KeyboardInterrupt:
            video.release()
            cv2.destroyAllWindows()
            del video
            self.plotter.move_to(0, 0, 16000)
            raise(KeyboardInterrupt)
        
        video.release()
        self.plotter.move_to(0, 0, 16000)
        
        idx = 0
        w, h = sorted([dist(*points[0], *points[1]), dist(*points[1], *points[2])])
        for i in range(1, len(points)):
            x = dist(*points[i], *points[(i + 1) % len(points)])
            y = dist(*points[i], *points[i - 1])
            if x == h and y == w:
                idx = i
                break
    
        points = points[[(idx + i) % len(points) for i in range(len(points))]] 
        sx, sy = points[0]
        mx, my = points[1]
        angle = get_angle(0, 1, mx - sx, my - sy)
        
        return angle, w, sx, sy, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def read_image_from_ax(self, ax):
        ax.axis("off")
        ax.figure.canvas.draw()
        buff = io.BytesIO()
        trans = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.bbox.transformed(trans)
        plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox)
        buff.seek(0)
        img = plt.imread(buff)
        return img
        
    def draw_trajectory(self, trajectory, image):
        ax = plt.subplot()
        image = np.full((*image.shape, 3), 255)
        ax.imshow(image.astype('uint8'), vmin=0)
        i = 0
        end = 0
        while i < len(trajectory):
            while i < len(trajectory) and trajectory[i, 0] != -1e9:
                i += 1
            i += 1
            end = i
            while i < len(trajectory) and trajectory[i, 0] != 1e9:
                i += 1
            ax.add_line(Line2D(trajectory[end:i, 1], image.shape[1] - trajectory[end:i, 0], lw=1, color='blue'))
        return self.read_image_from_ax(ax)
    
    def get_trajectory(self, img, crop=False, show=False):
        """
        Рисует изображение.
        
        Параметры:
        img (np.ndarray): Изображение.
        crop (bool): Обрезать изображение.
        show (bool): Показать изображение.
        """
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        img = img[::-1]
        #img = rotate(img, 90, mode='edge')[::-1]
        print('getting colors..')
        img = get_colors(img, crop)
        print('got colors')
        
        print('getting trajectory...')
        trajectory = self.get_raw_trajectory(img)
        print('got trajectory')
        # print(trajectory)
        trajectory = np.array(trajectory)
        drawed = self.draw_trajectory(trajectory, img)
        if show:
            return trajectory, drawed, img
        
        return trajectory

    def shift_trajectory(self, trajectory, img):
        angle, w, sx, sy, frame = self.compute_angle()
        index = np.any(np.abs(trajectory) != [1e9, 1e9], axis=1)
        
        k = w / max(img.shape) / 1.1
        test_shift = shift_coords(sx, sy, pi / 2 - angle, trajectory[index], k)
        if np.max(test_shift) > max(frame.shape): angle = pi/2 + angle
        else: angle = pi / 2 - angle
        trajectory[index] = shift_coords(sx, sy, angle, trajectory[index], k)
        # print(trajectory.tolist())
        trajectory = trajectory[:, [1, 0]]
        return trajectory
    
    def start(self, image: np.ndarray):
        process = Process(target=self.work, args=(image, ))
        process.start()

    def work(self, image):
        self.lib = TrajectoryLib(
            self.cpp_path, 
            self.lib_folder, 
            self.cuda_version, self.gpu_arch
        )
        self.plotter = Plotter(self.video_id, self.stop, port=self.port, baudrate=self.baudrate)
        
        trajectory, image, base_img = self.get_trajectory(image, show=True)
        self.images_queue.put(image)
        while not self.flag_start.value: continue
        if self.flag_start.value == 2:
            return
        trajectory = self.shift_trajectory(trajectory, base_img)
        self.plotter.calibrate_servo()
        self.plotter.plot_trajectory(trajectory)
        self.flag_end.value = 1