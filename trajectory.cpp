#include <math.h>
#include <vector>
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
typedef vector <vector <vector <ld>>> vvvd;
typedef vector <vector <ld>> vvd;
typedef vector <vector <int>> vvi;
typedef vector <int> vi;
typedef vector <ld> vd;
typedef vector <pair <int, int>> vc;

extern "C"
{
    bool eqvdvd(const vd &a, const vd &b) {
        if (a.size() != b.size()) return false; 
        return equal(execution::par_unseq, a.begin(), a.end(), b.begin());
    }

    bool neqvdvd (const vd &a, const vd &b)
    {
        return !eqvdvd(a, b);
    }

    bool eqvvivvi (const vvi &a, const vvi &b)
    {
        if (a.size() != b.size() || a[0].size() != b[0].size()) return false; 

        for (int x = 0; x < a.size(); ++x) {
            auto [it1, it2] = mismatch(execution::par_unseq, a[x].begin(), a[x].end(), b[x].begin());
            if (it1 != a[x].end()) return false; 
        }
        return true;
    }

    bool neqvii(vvi a, int b) {
        return any_of(a.begin(), a.end(), [=](const std::vector<int> row) {
            return any_of(execution::par_unseq, row.begin(), row.end(), [=](int elem) {
                return elem != b;
            });
        });
    }

    vd subvdvd(const vd &a, const vd &b) {
        vd ans(a.size());
        transform(execution::par_unseq, a.begin(), a.end(), b.begin(), ans.begin(), [](ld x, ld y) {
            return x - y;
        });
        return ans;
    }

    vvd subvdvvd (const vd a, const vvd b)
    {
        vvd ans(b.size());
        for (int i = 0; i < b.size(); i++) ans[i] = subvdvd(a, b[i]);
        return ans;
    }

    vvi vand(const vvi &a, const vvi &b) {
        vvi ans(a.size(), vector<int>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            transform(execution::par_unseq, a[i].begin(), a[i].end(), b[i].begin(), ans[i].begin(), [](int x, int y) {
                return x && y;
            });
        }
        return ans;
    }

    vvi vnot(const vvi &a)
    {
        vvi ans(a.size(), vi(a[0].size()));
        for (int i = 0; i < a.size(); i++)
        {
            transform(execution::par_unseq, a[i].begin(), a[i].end(), ans[i].begin(), [](int x) {return !x;});
        }
        return ans;
    }

    vd absvd(const vd &x)
    {
        vd ans(x.size());
        transform(execution::par_unseq, x.begin(), x.end(), ans.begin(), [](ld a) {return abs(a);});
        return ans;
    }

    ld sumvd(const vd &x)
    {
        ld ans = reduce(execution::par_unseq, x.begin(), x.end(), 0);
        return ans;
    }

    vd sumvvd(const vvd &x)
    {
        vd ans(x.size());
        for (int i = 0; i < x.size(); i++) ans[i] = sumvd(x[i]);
        return ans;
    }

    vvi inside_or(vector <vc> &arr)
    {
        vvi ans(arr.size(), vi(ans[0].size(), 0));
        for (int x = 0; x < arr.size(); x++)
        {
            transform(execution::par_unseq, arr[x].begin(), arr[x].end(), ans[x].begin(), [](pair <int, int> &cords) {return cords.first || cords.second;});
        }
        return ans;
    }

    vc var = {make_pair(-1, 0), make_pair(0, -1), make_pair(0, 1), make_pair(1, 0), make_pair(-1, -1), make_pair(-1, 1), make_pair(1, -1), make_pair(1, 1)};

    bool in_image(int x, int y, pair <int, int> shape)
    {
        return x >= 0 && y >= 0 && x < shape.first && y < shape.second;
    }

    int argmin(const vd &arr)
    {
        int idx = -1;
        ld mx = -1e9;
        for (int i = 0; i < arr.size(); i++)
        {
            if (arr[i] > mx)
            {
                mx = arr[i];
                idx = i;
            }
        }
        return idx;
    }

    vvd cmark(vvd &img, vvd &itsa)
    {
        vvd nimg(img.size(), vd (3));
        vd zrs = {0, 0, 0};
        for (int i = 0; i < img.size(); i++)
        {
            vd px = img[i];
            if (px == zrs) nimg[i] = itsa[argmin(absvd(sumvvd(subvdvvd(px, itsa))))];
        }
        return nimg;
    }

    void cfill(int x, int y, vvi &vis, vvvd &img)
    {
        vvi vis1 = vis;
        vis[x][y] = 1;
        deque <pair <vi, vd>> ln = {{{x, y, 0}, {0, 0, 0}}};
        int xn, yn;
        vd clr(3);
        bool flg;
        vd zrs = {0, 0, 0};
        pair <int, int> shp = {img.size(), img[0].size()};
        while (ln.size() != 0)
        {
            int sz = ln.size();
            for (int i = 0; i < sz; i++)
            {
                pair <vi, vd> val = ln.front();
                x = val.first[0];
                y = val.first[1];
                flg = val.first[2];
                clr = val.second;
                ln.pop_front();
                for (int j = 0; j < var.size(); j++)
                {
                    int dx = var[j].first, dy = var[j].second;
                    xn = x + dx;
                    yn = y + dy;
                    if (in_image(xn, yn, shp))
                    {
                        if (flg)
                        {
                            img[xn][yn] = clr;
                            if (!vis1[xn][yn])
                            {
                                ln.push_back({{xn, yn, flg}, clr});
                                vis1[xn][yn] = 1;
                            }
                        }
                        else if (img[xn][yn] != zrs)
                        {
                            clr = img[xn][yn];
                            img[x][y] = clr;
                            flg = 1;
                            ln.push_back({{xn, yn, flg}, clr});
                        }
                        else if (!vis[xn][yn])
                        {
                            vis[xn][yn] = 1;
                            ln.push_back({{xn, yn, flg}, clr});
                        }
                    }
                }
            }
        }
    }

    vc nonzero(vvi &vec)
    {
        vc ans;
        for (int x = 0; x < vec.size(); x++)
        {
            for (int y = 0; y < vec[0].size(); y++)
            {
                if (vec[x][y]) ans.push_back(make_pair(x, y));
            }
        }
        return ans;
    }

    bool check_near(int x, int y, vc &deltas, vvi &bimage)
    {
        int xn, yn, dtx, dty;
        pair <int, int> shp = {bimage.size(), bimage[0].size()};
        for (int i = 0; i < deltas.size(); i++)
        {
            dtx = deltas[i].first;
            dty = deltas[i].second;
            xn = x + dtx;
            yn = y + dty;
            if (in_image(xn, yn, shp))
            {
                if (!bimage[xn][yn]) return 0;
            }
            else return 0;
        }
        return 1;
    }

    pair <int, int> random_point(vvi &bimage, vc &deltas)
    {
        int x = -1, y = -1;
        vc nz = nonzero(bimage);
        int cnt = 0;
        while (cnt < nz.size() && !check_near(x, y, deltas, bimage))
        {
            x = nz[cnt].first;
            y = nz[cnt].second;
            cnt++;
        }
        if (x == -1) return make_pair(x, y);
        if (!check_near(x, y, deltas, bimage)) return make_pair(-1, -1);
        return make_pair(x, y);
    }

    pair <int, int> random_pointf(vvi &bimage, vc &deltas, vvi &filter)
    {
        int x = -1, y = -1;
        vc nz = nonzero(bimage);
        int cnt = 0;
        pair <int, int> shp = make_pair(filter.size(), filter[0].size());
        while (cnt < nz.size())
        {
            if (in_image(x, y, shp))
            {
                if (check_near(x, y, deltas, filter)) break;
            }
            x = nz[cnt].first;
            y = nz[cnt].second;
            cnt++;
        }
        return make_pair(x, y);
    }

    void get_deltas(vc &deltas2, vc &mxdeltas, vc &deltas, int d)
    {
        deltas.push_back(make_pair(0, 0));
        for (int x = -d - 2; x <= d + 2; x++)
        {
            for (int y = -d - 2; y <= d + 2; y++)
            {
                if (x == 0 && y == 0) continue;
                if (sqrt((ld) (x * x + y * y)) <= (d / 2 + 0.5))
                {
                    deltas.push_back(make_pair(x, y));
                    if (abs(sqrt((ld) (x * x + y * y)) - d / 2) <= 0.5) 
                        mxdeltas.push_back(make_pair(x, y));
                }
                if (abs(sqrt((ld) (x * x + y * y)) - (d + 2)) <= 0.5) 
                    deltas2.push_back(make_pair(x, y));
            }
        }
    }

    vc get_path(ld bx, ld by, ld tx, ld ty)
    {
        vc ans;
        ld vx, vy;
        vx = tx - bx;
        vy = ty - by;
        ld len = sqrt(vx * vx + vy * vy);
        if (len == 0)
        {
            ans.push_back(make_pair(bx, by));
            return ans;
        }
        vx /= len;
        vy /= len;
        ld x = bx, y = by;
        ld i = 0;
        bool cond = 1;
        while (cond)
        {
            ans.push_back(make_pair(x, y));
            x = bx + i * vx;
            y = by + i * vy;
            i++;
            cond = !((abs(x - tx) < 1e-6) && (abs(y - ty) < 1e-6));
            if (tx - x != 0) cond = cond && ((x - bx) / (tx - x) >= 0);
            if (ty - y != 0) cond = cond && ((y - by) / (ty - y) >= 0);
        }
        ans.push_back(make_pair(tx, ty));
        return ans;
    }

    void save(vvi &image)
    {
        int x, y, r, g, b;
        int w = image.size(), h = image[0].size();
        FILE *f;
        unsigned char *img = NULL;
        int filesize = 54 + 3 * w * h;  //w is your image width, h is image height, both int

        img = (unsigned char *)malloc(3 * w * h);
        memset(img, 0, 3 * w * h);

        for(int i = 0; i < w; i++)
        {
            for(int j = 0; j < h; j++)
            {
                x = i; y = (h - 1) - j;
                r = image[i][j] * 255;
                g = image[i][j] * 255;
                b = image[i][j] * 255;
                if (r > 255) r = 255;
                if (g > 255) g = 255;
                if (b > 255) b = 255;
                img[(x + y * w) * 3 + 2] = (unsigned char)(r);
                img[(x + y * w) * 3 + 1] = (unsigned char)(g);
                img[(x + y * w) * 3 + 0] = (unsigned char)(b);
            }
        }

        unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
        unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
        unsigned char bmppad[3] = {0,0,0};

        bmpfileheader[ 2] = (unsigned char)(filesize    );
        bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
        bmpfileheader[ 4] = (unsigned char)(filesize>>16);
        bmpfileheader[ 5] = (unsigned char)(filesize>>24);

        bmpinfoheader[ 4] = (unsigned char)(       w    );
        bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
        bmpinfoheader[ 6] = (unsigned char)(       w>>16);
        bmpinfoheader[ 7] = (unsigned char)(       w>>24);
        bmpinfoheader[ 8] = (unsigned char)(       h    );
        bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
        bmpinfoheader[10] = (unsigned char)(       h>>16);
        bmpinfoheader[11] = (unsigned char)(       h>>24);

        f = fopen("images/img.bmp", "wb");
        fwrite(bmpfileheader, 1, 14, f);
        fwrite(bmpinfoheader, 1, 40, f);
        for(int i = 0; i < h; i++)
        {
            fwrite(img + (w * (h - i - 1) * 3), 3, w, f);
            fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
        }

        free(img);
        fclose(f);
    }

    void get_trajectory(vvi &bimage, int d, int sx, int sy, vc &ans)
    {
        vc deltas2, mxdeltas, deltas;
        get_deltas(deltas2, mxdeltas, deltas, d);

        vvi filter = bimage;

        int x, y, xn, yn, xp, yp, it, xnn, ynn;
        bool change, flagfill = 1;
        
        pair <int, int> res = random_point(bimage, mxdeltas);
        x = res.first;
        y = res.second;

        ans.push_back(make_pair(x, y));
        ans.push_back(make_pair(-1e9, -1e9));

        xp = 0;
        yp = 0;
        it = 0;

        vc path;
        pair <int, int> shp = {bimage.size(), bimage[0].size()};

        int zero = 0;
        while (neqvii(bimage, zero))
        {
            // if (it % 1000 == 0)
            // {
            //     save(bimage);
            //     sleep(2);
            // }
            // cout << nonzero(bimage).size() << '\n';
            change = 0;
            flagfill = 1;
            for (pair <int, int> dt : deltas2)
            {
                xn = x + dt.first; yn = y + dt.second;
                if (in_image(xn, yn, shp))
                {
                    if (bimage[xn][yn] && check_near(xn, yn, mxdeltas, bimage))
                    {
                        x = xn; y = yn;
                        ans.push_back(make_pair(x + sx, y + sy));
                        change = 1;
                        break;
                    }
                }
            }

            if (!change)
            {
                res = random_point(bimage, mxdeltas);
                x = res.first; y = res.second;
                if (x == -1)
                {
                    res = random_pointf(bimage, mxdeltas, filter);
                    x = res.first; y = res.second;
                }
                if (bimage[x][y])
                {
                    ans.push_back(make_pair(1e9, 1e9));
                    ans.push_back(make_pair(x + sx, y + sy));
                    ans.push_back(make_pair(-1e9, -1e9));
                    ans.push_back(make_pair(x + sx, y + sy));
                }    
                flagfill = 0;
            }

            if (flagfill && it)
            {
                path = get_path(xp, yp, x, y);
                for (pair <int, int> xyn : path)
                {
                    xn = xyn.first, yn = xyn.second;
                    
                    for (pair <int, int> dt : deltas)
                    {
                        xnn = xn + dt.first; ynn = yn + dt.second;
                        if (in_image(xnn, ynn, shp)) bimage[xnn][ynn] = 0;
                    }
                }
            }
            else
            {
                for (pair <int, int> dt : deltas)
                {
                    xn = x + dt.first; yn = y + dt.second;
                    if (in_image(xn, yn, shp)) bimage[xn][yn] = 0;
                }
            }
            xp = x;
            yp = y;
            it++;
        }
    }

    bool check_poly(vc &cords, ld a, ld b)
    {
        ld s = 0;
        for (pair <int, int> pnt : cords)
            s = max(s, abs(a * pnt.first + b - pnt.second));
        return s < 3;
    }

    ld mean(vi &vec)
    {
        ld s = 0;
        for (int e : vec) s += e;
        return s / vec.size();
    }

    ld variance(vi &lst, ld mn)
    {
        ld s = 0;
        for (int e : lst)
            s += (e - mn) * (e - mn);
        return s / lst.size();
    }

    ld variancexy(vi &x, vi &y, ld xm, ld ym)
    {
        ld s = 0;
        for (int i = 0; i < x.size(); i++)
            s += (x[i] - xm) * (y[i] - ym);
        return s / x.size();
    }

    pair <bool, pair <ld, ld>> get_line(vc &pnts)
    {
        vi x, y;
        for (pair <int, int> pnt : pnts)
        {
            x.push_back(pnt.first);
            y.push_back(pnt.second);
        }

        ld xm = mean(x), ym = mean(y);
        ld sx = variance(x, xm), sy = variance(y, ym);
        ld sxy = variancexy(x, y, xm, ym);

        if (sxy == 0) return {0, make_pair(0, 0)};
        ld a = (sy - sx + sqrt((sy - sx) * (sy - sx) + 4 * sxy * sxy)) / (2 * sxy);
        ld b = ym - a * xm;
        return {1, make_pair(a, b)};
    }

    vc approximate(vc &cords)
    {
        int i = 2; 
        vc ncords = {cords[0], cords[1]};
        vc now;
        ld an, bn;
        while (i < cords.size())
        {
            if (cords[i].first == 1e9) 
            {
                if (now.size() > 1)
                {
                    ncords.push_back(now[0]);
                    ncords.push_back(now.back());
                }
                ncords.push_back(cords[i]);
                i++;
                now.clear();
                continue;
            }
            else if (cords[i].first == -1e9) 
            {
                if (!now.empty())
                {
                    for (pair <int, int> e: now) ncords.push_back(e);
                } 
                ncords.push_back(cords[i]);
                now.clear();
                i++;
                continue;
            }
            
            now.push_back(cords[i]);
            if (now.size() < 2)
            {
                i++;
                continue;
            }

            pair <bool, pair <ld, ld>> res = get_line(now);
            if (res.first)
            {
                an = res.second.first;
                bn = res.second.second;
            }
            else
            {
                i++;
                continue;
            }

            if (!check_poly(now, an, bn))
            {
                if (now.size() > 1)
                {
                    ncords.push_back(now[0]);
                    ncords.push_back(now[now.size() - 2]);
                }
                now = {cords[i]};
            }
            i++;
        }
        if (now.size() > 1)
        {
            ncords.push_back(now[0]);
            ncords.push_back(now.back());
        }
        return ncords;
    }

    void remove_dublicates(vc &x)
    {
        int i = 0; 
        while (i < x.size() - 3)
        {
            if (x[i].first == 1e9 && x[i + 2].first == -1e9 && x[i + 3].first == 1e9)
            {
                x.erase(x.begin() + i);
                x.erase(x.begin() + i);
                x.erase(x.begin() + i);
            }
            else i++;
        }
    }

    void pointer2vvvd(ld* pointer, size_t x, size_t y, size_t z, vvvd &ans)
    {
        for (int a = 0; a < x; a++)
        {
            for (int b = 0; b < y; b++)
            {
                for (int c = 0; c < z; c++) ans[a][b][c] = pointer[a * y + b * z + c];
            }
        }
    }

    void vvvd2pointer(vvvd &vec, ld* pointer)
    {
        for (int a = 0; a < vec.size(); a++)
        {
            for (int b = 0; b < vec[0].size(); b++)
            {
                for (int c = 0; c < vec[0][0].size(); c++) pointer[a * vec.size() + b * vec[0].size() + c] = vec[a][b][c];
            }
        }
    }

    void pointer2vvd(ld* pointer, size_t n, size_t m, vvd &ans)
    {
        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < m; y++) ans[x][y] = pointer[x * m + y];
        }
    }

    void vvd2pointer(vvd &vec, ld* ans)
    {
        for (int x = 0; x < vec.size(); x++)
        {
            for (int y = 0; y < vec[0].size(); y++) ans[x * vec[0].size() + y] = vec[x][y];
        }
    }

    void pointer2vvi(int *pointer, size_t n, size_t m, vvi &ans)
    {
        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < m; y++) ans[x][y] = pointer[x * m + y];
        }
    }

    void vvi2pointer(vvi &vec, int* pointer)
    {
        for (int x = 0; x < vec.size(); x++)
        {
            for (int y = 0; y < vec[0].size(); y++) pointer[x * vec[0].size() + y] = vec[x][y];
        }
    }

    void pointer2vc(int *pointer, size_t n, vc &ans)
    {
        for (int i = 0; i < n; i++) 
        {
            ans[i].first = pointer[i * 2];
            ans[i].second = pointer[i * 2 + 1]; 
        }
    }

    int* vc2pointer(vc &vec)
    {
        int* ans = new int[vec.size() * 2 + 1];
        ans[0] = vec.size();
        for (int i = 0; i < vec.size(); i++)
        {
            ans[i * 2 + 1] = vec[i].first;
            ans[i * 2 + 2] = vec[i].second;
        }
        return ans;
    }

    int* pcompute_image(int* pointer, size_t n, size_t m, int d, int sx, int sy)
    {
        vvi image(n, vi(m));
        pointer2vvi(pointer, n, m, image);
        vc trajectory;
        get_trajectory(image, d, sx, sy, trajectory);
        trajectory = approximate(trajectory);
        remove_dublicates(trajectory);
        return vc2pointer(trajectory);
    }

    void pmark(ld* imgp, ld* clrsp, size_t imgh, size_t imgw, size_t clrsh, size_t clrsw)
    {
        vvd img(imgh, vd(imgw)), clrs(clrsh, vd(clrsw));
        pointer2vvd(imgp, imgh, imgw, img);
        pointer2vvd(clrsp, clrsh, clrsw, clrs);
        img = cmark(img, clrs);
        vvd2pointer(img, imgp);
    }

    void pfill(int x, int y, int* visp, ld* imgp, size_t imgh, size_t imgw, size_t chnls)
    {
        vvi vis(imgh, vi(imgw));
        vvvd img(imgh, vvd(imgw, vd(chnls)));
        pointer2vvvd(imgp, imgh, imgw, chnls, img);
        pointer2vvi(visp, imgh, imgw, vis);
        cfill(x, y, vis, img);
        vvvd2pointer(img, imgp);
        vvi2pointer(vis, visp);
    }

    void cleanup(int* pntr)
    {
        delete[] pntr;
    }
}
