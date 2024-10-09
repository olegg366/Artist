#include <vector>
#include <math.h>
#include <deque>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <unistd.h>
#include <iomanip>
#include <execution>
#include <algorithm>
#include <numeric>

using namespace std;

typedef double ld;
typedef vector<vector<vector<ld>>> Matrix3D;
typedef vector<vector<ld>> Matrix2D;
typedef vector<vector<int>> IntMatrix2D;
typedef vector<int> IntVector;
typedef vector<ld> Vector;
typedef vector<pair<int, int>> PointVector;

extern "C"
{
    // Проверяет, есть ли в матрице элементы, не равные заданному значению
    bool has_elements_not_equal(const IntMatrix2D &matrix, int value) {
        return any_of(matrix.begin(), matrix.end(), [=](const vector<int> &row) {
            return any_of(execution::par, row.begin(), row.end(), [=](int elem) {
                return elem != value;
            });
        });
    }

    // Вычитает два вектора
    Vector subtract_vectors(const Vector &a, const Vector &b) {
        Vector result(a.size());
        transform(execution::par, a.begin(), a.end(), b.begin(), result.begin(), [](ld x, ld y) {
            return x - y;
        });
        return result;
    }

    // Вычитает вектор из каждой строки матрицы
    Matrix2D subtract_vector_from_matrix(const Vector &vector, const Matrix2D &matrix) {
        Matrix2D result(matrix.size());
        for (int i = 0; i < matrix.size(); i++) {
            result[i] = subtract_vectors(vector, matrix[i]);
        }
        return result;
    }

    // Возвращает вектор абсолютных значений элементов вектора
    Vector absolute_values(const Vector &vector) {
        Vector result(vector.size());
        transform(execution::par, vector.begin(), vector.end(), result.begin(), [](ld a) {
            return abs(a);
        });
        return result;
    }

    // Возвращает сумму элементов вектора
    ld sum_of_vector(const Vector &vector) {
        ld result = reduce(execution::par, vector.begin(), vector.end(), 0);
        return result;
    }

    // Возвращает вектор сумм элементов каждой строки матрицы
    Vector sum_of_matrix_rows(const Matrix2D &matrix) {
        Vector result(matrix.size());
        for (int i = 0; i < matrix.size(); i++) {
            result[i] = sum_of_vector(matrix[i]);
        }
        return result;
    }

    // Вектор смещений для обхода соседних точек
    PointVector offsets = {
        make_pair(-1, 0), make_pair(0, -1), make_pair(0, 1), make_pair(1, 0),
        make_pair(-1, -1), make_pair(-1, 1), make_pair(1, -1), make_pair(1, 1)
    };

    // Проверяет, находится ли точка внутри изображения
    bool is_point_in_image(int x, int y, pair<int, int> shape) {
        return x >= 0 && y >= 0 && x < shape.first && y < shape.second;
    }

    // Возвращает индекс минимального элемента в векторе
    int index_of_min_element(const Vector &vector) {
        int index = -1;
        ld min_value = 1e9;
        for (int i = 0; i < vector.size(); i++) {
            if (vector[i] < min_value) {
                min_value = vector[i];
                index = i;
            }
        }
        return index;
    }

    // Маркирует изображение на основе цветовых шаблонов
    Matrix2D mark_image(Matrix2D &image, Matrix2D &color_patterns) {
        Matrix2D new_image(image.size(), Vector(3));
        Vector zero_vector = {0, 0, 0};
        for (int i = 0; i < image.size(); i++) {
            Vector pixel = image[i];
            if (pixel == zero_vector) {
                new_image[i] = color_patterns[index_of_min_element(absolute_values(sum_of_matrix_rows(subtract_vector_from_matrix(pixel, color_patterns))))];
            }
        }
        return new_image;
    }

    // Заполняет область на изображении
    void fill_area(int x, int y, IntMatrix2D &visited, Matrix3D &image) {
        IntMatrix2D visited_copy = visited;
        visited[x][y] = 1;
        deque<pair<IntVector, Vector>> queue = {{{x, y, 0}, {0, 0, 0}}};
        int new_x, new_y;
        Vector color(3);
        bool flag;
        Vector zero_vector = {0, 0, 0};
        pair<int, int> shape = {image.size(), image[0].size()};
        while (!queue.empty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                pair<IntVector, Vector> value = queue.front();
                x = value.first[0];
                y = value.first[1];
                flag = value.first[2];
                color = value.second;
                queue.pop_front();
                for (int j = 0; j < offsets.size(); j++) {
                    int dx = offsets[j].first, dy = offsets[j].second;
                    new_x = x + dx;
                    new_y = y + dy;
                    if (is_point_in_image(new_x, new_y, shape)) {
                        if (flag) {
                            image[new_x][new_y] = color;
                            if (!visited_copy[new_x][new_y]) {
                                queue.push_back({{new_x, new_y, flag}, color});
                                visited_copy[new_x][new_y] = 1;
                            }
                        } else if (image[new_x][new_y] != zero_vector) {
                            color = image[new_x][new_y];
                            image[x][y] = color;
                            flag = 1;
                            queue.push_back({{new_x, new_y, flag}, color});
                        } else if (!visited[new_x][new_y]) {
                            visited[new_x][new_y] = 1;
                            queue.push_back({{new_x, new_y, flag}, color});
                        }
                    }
                }
            }
        }
    }

    // Возвращает вектор координат ненулевых элементов матрицы
    PointVector get_nonzero_points(const IntMatrix2D &matrix) {
        PointVector result;
        for (int x = 0; x < matrix.size(); x++) {
            for (int y = 0; y < matrix[0].size(); y++) {
                if (matrix[x][y]) {
                    result.push_back(make_pair(x, y));
                }
            }
        }
        return result;
    }

    // Проверяет, есть ли вокруг точки ненулевые элементы
    bool check_nearby_points(int x, int y, const PointVector &deltas, const IntMatrix2D &binary_image) {
        int new_x, new_y, dx, dy;
        pair<int, int> shape = {binary_image.size(), binary_image[0].size()};
        for (int i = 0; i < deltas.size(); i++) {
            dx = deltas[i].first;
            dy = deltas[i].second;
            new_x = x + dx;
            new_y = y + dy;
            if (is_point_in_image(new_x, new_y, shape)) {
                if (!binary_image[new_x][new_y]) {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }

    // Возвращает случайную точку, удовлетворяющую условиям
    pair<int, int> get_random_point(const IntMatrix2D &binary_image, const PointVector &deltas) {
        int x = -1, y = -1;
        PointVector nonzero_points = get_nonzero_points(binary_image);
        int count = 0;
        while (count < nonzero_points.size() && !check_nearby_points(x, y, deltas, binary_image)) {
            x = nonzero_points[count].first;
            y = nonzero_points[count].second;
            count++;
        }
        if (x == -1) {
            return make_pair(x, y);
        }
        if (!check_nearby_points(x, y, deltas, binary_image)) {
            return make_pair(-1, -1);
        }
        return make_pair(x, y);
    }

    // Возвращает случайную точку, удовлетворяющую условиям с фильтром
    pair<int, int> get_random_point_with_filter(const IntMatrix2D &binary_image, const PointVector &deltas, const IntMatrix2D &filter) {
        int x = -1, y = -1;
        PointVector nonzero_points = get_nonzero_points(binary_image);
        int count = 0;
        pair<int, int> shape = make_pair(filter.size(), filter[0].size());
        while (count < nonzero_points.size()) {
            if (is_point_in_image(x, y, shape)) {
                if (check_nearby_points(x, y, deltas, filter)) {
                    break;
                }
            }
            x = nonzero_points[count].first;
            y = nonzero_points[count].second;
            count++;
        }
        return make_pair(x, y);
    }

    // Генерирует вектор смещений для обхода точек
    void generate_deltas(PointVector &deltas2, PointVector &max_deltas, PointVector &deltas, int distance) {
        deltas.push_back(make_pair(0, 0));
        for (int x = -distance - 2; x <= distance + 2; x++) {
            for (int y = -distance - 2; y <= distance + 2; y++) {
                if (x == 0 && y == 0) {
                    continue;
                }
                if (sqrt((ld)(x * x + y * y)) <= (distance / 2 + 0.5)) {
                    deltas.push_back(make_pair(x, y));
                    if (abs(sqrt((ld)(x * x + y * y)) - distance / 2) <= 0.5) {
                        max_deltas.push_back(make_pair(x, y));
                    }
                }
                if (abs(sqrt((ld)(x * x + y * y)) - (distance + 2)) <= 0.5) {
                    deltas2.push_back(make_pair(x, y));
                }
            }
        }
    }

    // Возвращает путь между двумя точками
    PointVector get_path(ld start_x, ld start_y, ld end_x, ld end_y) {
        PointVector result;
        ld vx, vy;
        vx = end_x - start_x;
        vy = end_y - start_y;
        ld length = sqrt(vx * vx + vy * vy);
        if (length == 0) {
            result.push_back(make_pair(start_x, start_y));
            return result;
        }
        vx /= length;
        vy /= length;
        ld x = start_x, y = start_y;
        ld i = 0;
        bool condition = true;
        while (condition) {
            result.push_back(make_pair(x, y));
            x = start_x + i * vx;
            y = start_y + i * vy;
            i++;
            condition = !((abs(x - end_x) < 1e-6) && (abs(y - end_y) < 1e-6));
            if (end_x - x != 0) {
                condition = condition && ((x - start_x) / (end_x - x) >= 0);
            }
            if (end_y - y != 0) {
                condition = condition && ((y - start_y) / (end_y - y) >= 0);
            }
        }
        result.push_back(make_pair(end_x, end_y));
        return result;
    }

    // Генерирует траекторию на основе бинарного изображения
    void generate_trajectory(IntMatrix2D &binary_image, int distance, int start_x, int start_y, PointVector &result) {
        PointVector deltas2, max_deltas, deltas;
        generate_deltas(deltas2, max_deltas, deltas, distance);

        IntMatrix2D filter = binary_image;

        int x, y, new_x, new_y, prev_x, prev_y, iteration, new_new_x, new_new_y;
        bool change, fill_flag = true;

        pair<int, int> random_point_result = get_random_point(binary_image, max_deltas);
        x = random_point_result.first;
        y = random_point_result.second;

        result.push_back(make_pair(x + start_x, y + start_y));
        result.push_back(make_pair(-1e9, -1e9));

        prev_x = 0;
        prev_y = 0;
        iteration = 0;

        PointVector path;
        pair<int, int> shape = {binary_image.size(), binary_image[0].size()};

        int zero = 0;
        while (has_elements_not_equal(binary_image, zero)) {
            change = false;
            fill_flag = true;
            for (pair<int, int> delta : deltas2) {
                new_x = x + delta.first;
                new_y = y + delta.second;
                if (is_point_in_image(new_x, new_y, shape)) {
                    if (binary_image[new_x][new_y] && check_nearby_points(new_x, new_y, deltas, binary_image)) {
                        x = new_x;
                        y = new_y;
                        result.push_back(make_pair(x + start_x, y + start_y));
                        change = true;
                        break;
                    }
                }
            }

            if (!change) {
                random_point_result = get_random_point(binary_image, deltas);
                x = random_point_result.first;
                y = random_point_result.second;
                if (x == -1) {
                    random_point_result = get_random_point_with_filter(binary_image, deltas, filter);
                    x = random_point_result.first;
                    y = random_point_result.second;
                }
                if (binary_image[x][y]) {
                    result.push_back(make_pair(1e9, 1e9));
                    result.push_back(make_pair(x + start_x, y + start_y));
                    result.push_back(make_pair(-1e9, -1e9));
                    result.push_back(make_pair(x + start_x, y + start_y));
                }
                fill_flag = false;
            }

            if (fill_flag && iteration) {
                path = get_path(prev_x, prev_y, x, y);
                for (pair<int, int> point : path) {
                    new_x = point.first;
                    new_y = point.second;

                    for (pair<int, int> delta : deltas) {
                        new_new_x = new_x + delta.first;
                        new_new_y = new_y + delta.second;
                        if (is_point_in_image(new_new_x, new_new_y, shape)) {
                            binary_image[new_new_x][new_new_y] = 0;
                        }
                    }
                }
            } else {
                for (pair<int, int> delta : deltas) {
                    new_x = x + delta.first;
                    new_y = y + delta.second;
                    if (is_point_in_image(new_x, new_y, shape)) {
                        binary_image[new_x][new_y] = 0;
                    }
                }
            }
            prev_x = x;
            prev_y = y;
            iteration++;
        }
    }

    // Проверяет, лежат ли точки на прямой
    bool check_points_on_line(const PointVector &points, ld a, ld b) {
        ld s = 0;
        for (pair<int, int> point : points) {
            s = max(s, abs(a * point.first + b - point.second));
        }
        return s < 0.5;
    }

    // Возвращает среднее значение вектора
    ld mean_of_vector(const IntVector &vector) {
        ld s = 0;
        for (int element : vector) {
            s += element;
        }
        return s / vector.size();
    }

    // Возвращает дисперсию вектора
    ld variance_of_vector(const IntVector &vector, ld mean) {
        ld s = 0;
        for (int element : vector) {
            s += (element - mean) * (element - mean);
        }
        return s / vector.size();
    }

    // Возвращает ковариацию двух векторов
    ld covariance_of_vectors(const IntVector &x, const IntVector &y, ld mean_x, ld mean_y) {
        ld s = 0;
        for (int i = 0; i < x.size(); i++) {
            s += (x[i] - mean_x) * (y[i] - mean_y);
        }
        return s / x.size();
    }

    // Возвращает параметры прямой, проходящей через точки
    pair<bool, pair<ld, ld>> get_line_parameters(const PointVector &points) {
        IntVector x, y;
        for (pair<int, int> point : points) {
            x.push_back(point.first);
            y.push_back(point.second);
        }

        ld mean_x = mean_of_vector(x);
        ld mean_y = mean_of_vector(y);
        ld variance_x = variance_of_vector(x, mean_x);
        ld variance_y = variance_of_vector(y, mean_y);
        ld covariance_xy = covariance_of_vectors(x, y, mean_x, mean_y);

        if (covariance_xy == 0) {
            return {false, make_pair(0, 0)};
        }
        ld a = (variance_y - variance_x + sqrt((variance_y - variance_x) * (variance_y - variance_x) + 4 * covariance_xy * covariance_xy)) / (2 * covariance_xy);
        ld b = mean_y - a * mean_x;
        return {true, make_pair(a, b)};
    }

    // Аппроксимирует траекторию
    PointVector approximate_trajectory(const PointVector &points) {
        int i = 2;
        PointVector new_points = {points[0], points[1]};
        PointVector current_points;
        ld a, b;
        while (i < points.size()) {
            if (points[i].first == 1e9) {
                if (current_points.size() > 1) {
                    new_points.push_back(current_points[0]);
                    new_points.push_back(current_points.back());
                }
                new_points.push_back(points[i]);
                i++;
                current_points.clear();
                continue;
            } else if (points[i].first == -1e9) {
                if (!current_points.empty()) {
                    for (pair<int, int> point : current_points) {
                        new_points.push_back(point);
                    }
                }
                new_points.push_back(points[i]);
                current_points.clear();
                i++;
                continue;
            }

            current_points.push_back(points[i]);
            if (current_points.size() < 2) {
                i++;
                continue;
            }

            pair<bool, pair<ld, ld>> result = get_line_parameters(current_points);
            if (result.first) {
                a = result.second.first;
                b = result.second.second;
            } else {
                i++;
                continue;
            }

            if (!check_points_on_line(current_points, a, b)) {
                if (current_points.size() > 1) {
                    new_points.push_back(current_points[0]);
                    new_points.push_back(current_points[current_points.size() - 2]);
                }
                current_points = {points[i]};
            }
            i++;
        }
        if (current_points.size() > 1) {
            new_points.push_back(current_points[0]);
            new_points.push_back(current_points.back());
        }
        return new_points;
    }

    // Удаляет дубликаты из траектории
    void remove_duplicates(PointVector &points) {
        int i = 0;
        while (i < points.size() - 3) {
            if (points[i].first == 1e9 && points[i + 2].first == -1e9 && points[i + 3].first == 1e9) {
                points.erase(points.begin() + i);
                points.erase(points.begin() + i);
                points.erase(points.begin() + i);
            } else {
                i++;
            }
        }
    }

    // Преобразует указатель на трехмерный массив в Matrix3D
    void pointer_to_matrix3d(ld* pointer, size_t x, size_t y, size_t z, Matrix3D &result) {
        for (int a = 0; a < x; a++) {
            for (int b = 0; b < y; b++) {
                for (int c = 0; c < z; c++) {
                    result[a][b][c] = pointer[a * y + b * z + c];
                }
            }
        }
    }

    // Преобразует Matrix3D в указатель на трехмерный массив
    void matrix3d_to_pointer(const Matrix3D &matrix, ld* pointer) {
        for (int a = 0; a < matrix.size(); a++) {
            for (int b = 0; b < matrix[0].size(); b++) {
                for (int c = 0; c < matrix[0][0].size(); c++) {
                    pointer[a * matrix.size() + b * matrix[0].size() + c] = matrix[a][b][c];
                }
            }
        }
    }

    // Преобразует указатель на двумерный массив в Matrix2D
    void pointer_to_matrix2d(ld* pointer, size_t n, size_t m, Matrix2D &result) {
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < m; y++) {
                result[x][y] = pointer[x * m + y];
            }
        }
    }

    // Преобразует Matrix2D в указатель на двумерный массив
    void matrix2d_to_pointer(const Matrix2D &matrix, ld* pointer) {
        for (int x = 0; x < matrix.size(); x++) {
            for (int y = 0; y < matrix[0].size(); y++) {
                pointer[x * matrix[0].size() + y] = matrix[x][y];
            }
        }
    }

    // Преобразует указатель на двумерный массив в IntMatrix2D
    void pointer_to_int_matrix2d(int* pointer, size_t n, size_t m, IntMatrix2D &result) {
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < m; y++) {
                result[x][y] = pointer[x * m + y];
            }
        }
    }

    // Преобразует IntMatrix2D в указатель на двумерный массив
    void int_matrix2d_to_pointer(const IntMatrix2D &matrix, int* pointer) {
        for (int x = 0; x < matrix.size(); x++) {
            for (int y = 0; y < matrix[0].size(); y++) {
                pointer[x * matrix[0].size() + y] = matrix[x][y];
            }
        }
    }

    // Преобразует PointVector в указатель на массив
    int* point_vector_to_pointer(const PointVector &vector) {
        int* result = new int[vector.size() * 2 + 1];
        result[0] = vector.size();
        for (int i = 0; i < vector.size(); i++) {
            result[i * 2 + 1] = vector[i].first;
            result[i * 2 + 2] = vector[i].second;
        }
        return result;
    }

    // Вычисляет траекторию на основе бинарного изображения
    int* compute_image_trajectory(int* pointer, size_t n, size_t m, int distance, int start_x, int start_y) {
        IntMatrix2D image(n, IntVector(m));
        pointer_to_int_matrix2d(pointer, n, m, image);
        PointVector trajectory;
        generate_trajectory(image, distance, start_x, start_y, trajectory);
        trajectory = approximate_trajectory(trajectory);
        remove_duplicates(trajectory);
        return point_vector_to_pointer(trajectory);
    }

    // Маркирует изображение на основе цветовых шаблонов
    void mark_image_with_colors(ld* image_pointer, ld* colors_pointer, size_t image_height, size_t image_width, size_t colors_height, size_t colors_width) {
        Matrix2D image(image_height, Vector(image_width));
        Matrix2D colors(colors_height, Vector(colors_width));
        pointer_to_matrix2d(image_pointer, image_height, image_width, image);
        pointer_to_matrix2d(colors_pointer, colors_height, colors_width, colors);
        image = mark_image(image, colors);
        matrix2d_to_pointer(image, image_pointer);
    }

    // Заполняет область на изображении
    void fill_image_area(int x, int y, int* visited_pointer, ld* image_pointer, size_t image_height, size_t image_width, size_t channels) {
        IntMatrix2D visited(image_height, IntVector(image_width));
        Matrix3D image(image_height, Matrix2D(image_width, Vector(channels)));
        pointer_to_matrix3d(image_pointer, image_height, image_width, channels, image);
        pointer_to_int_matrix2d(visited_pointer, image_height, image_width, visited);
        fill_area(x, y, visited, image);
        matrix3d_to_pointer(image, image_pointer);
        int_matrix2d_to_pointer(visited, visited_pointer);
    }

    // Освобождает память, выделенную под указатель
    void cleanup(int* pointer) {
        delete[] pointer;
    }
}