import cv2
import numpy as np



# Пример использования
if __name__ == "__main__":
    # Исходный контур с неправильным порядком вершин
    vertices = np.array([[100, 100], [300, 300], [100, 300], [300, 100]], dtype=np.int32)
    
    # Исправляем порядок вершин
    corrected_vertices = reorder_quadrilateral_vertices(vertices)
    
    # Создаем изображение для отображения результата
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Рисуем исходный контур
    cv2.drawContours(img, [vertices.reshape((-1, 1, 2))], -1, (0, 255, 0), 2)
    
    # Рисуем исправленный контур
    cv2.drawContours(img, [corrected_vertices.reshape((-1, 1, 2))], -1, (0, 0, 255), 2)
    
    # Отображаем изображение
    cv2.imshow("Quadrilateral with Corrected Order", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()