#include <math.h>
#include <vector>
#include <deque>
#include <iostream>

using namespace std;

typedef long double ld;
typedef vector <vector <vector <ld>>> vvvd;
typedef vector <vector <ld>> vvd;
typedef vector <vector <int>> vvi;
typedef vector <int> vi;
typedef vector <ld> vd;
typedef vector <pair <int, int>> vc;

extern "C"
{    
    bool eqvdvd (vd &a, vd &b)
    {
        for (int i = 0; i < a.size(); i++)
        {
            if (a[i] != b[i]) return 0;
        }
        return 1;
    }

    bool neqvdvd (vd &a, vd &b)
    {
        return !(a == b);
    }

    bool eqvvivvi (vvi &a, vvi &b)
    {
        for (int x = 0; x < a.size(); x++)
        {
            for (int y = 0; y < a[0].size(); y++)
            {
                if (a[x][y] != b[x][y]) return 0;
            }
        }
        return 1;
    }

    bool neqvii (vvi &a, const int &b)
    {
        for (int x = 0; x < a.size(); x++)
        {
            for (int y = 0; y < a[0].size(); y++)
            {
                if (a[x][y] != b) return 1;
            }
        }
        return 0;
    }

    vd subvdvd (vd &a, vd &b)
    {
        vd ans(a.size());
        for (int i = 0; i < a.size(); i++) ans[i] = a[i] - b[i];
        return ans;
    }

    vvd subvdvvd (vd &a, vvd &b)
    {
        vvd ans(b.size());
        for (int i = 0; i < b.size(); i++) ans[i] = subvdvd(a, b[i]);
        return ans;
    }

    vvi vand (vvi &a, vvi &b)
    {
        vvi ans(a.size(), vi(a[0].size()));
        for (int x = 0; x < a.size(); x++)
        {
            for (int y = 0; y < a[0].size(); y++) ans[x][y] = a[x][y] && b[x][y];
        }
        return ans;
    }

    vvi vnot(vvi &a)
    {
        vvi ans(a.size(), vi(a[0].size()));
        for (int x = 0; x < a.size(); x++)
        {
            for (int y = 0; y < a[0].size(); y++) ans[x][y] = !a[x][y];
        }
        return ans;
    }

    vd absvd(vd x)
    {
        vd ans(x.size());
        for (int i = 0; i < x.size(); i++) ans[i] = abs(x[i]);
        return ans;
    }

    ld sumvd(vd x)
    {
        ld ans = 0;
        for (int i = 0; i < x.size(); i++) ans += x[i];
        return ans;
    }

    vd sumvvd(vvd x)
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
            for (int y = 0; y < arr[0].size(); y++) ans[x][y] = arr[x][y].first || arr[x][y].second;
        }
        return ans;
    }

    vc var = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

    bool in_image(int x, int y, pair <int, int> shape)
    {
        return x >= 0 && y >= 0 && x < shape.first && y < shape.second;
    }

    int argmin(vd arr)
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

    vvd mark(vvd img, vvd itsa)
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

    pair <vvi, vvvd> fill(int x, int y, vvi &vis, vvvd &img)
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
        return {vis, img};
    }

    vc nonzero(vvi vec)
    {
        vc ans;
        for (int x = 0; x < vec.size(); x++)
        {
            for (int y = 0; y < vec[0].size(); y++)
            {
                if (vec[x][y]) ans.push_back({x, y});
            }
        }
        return ans;
    }

    bool check_near(int x, int y, vc deltas, vvi bimage)
    {
        int xn, yn, dtx, dty;
        for (int i = 0; i < deltas.size(); i++)
        {
            dtx = deltas[i].first;
            dty = deltas[i].second;
            xn = x + dtx;
            yn = y + dty;
            if (in_image(xn, yn, {bimage.size(), bimage[0].size()}))
            {
                if (!bimage[xn][yn]) return 0;
            }
        }
        return 1;
    }

    pair <int, int> random_point(vvi bimage, vc deltas)
    {
        int x, y;
        vc nz = nonzero(bimage);
        int cnt = nz.size();
        x = nz[0].first;
        y = nz[0].second;
        while (cnt >= 0 && !check_near(x, y, deltas, bimage))
        {
            cnt--;
            x = nz[cnt].first;
            y = nz[cnt].second;
        }
        if (!check_near(x, y, deltas, bimage)) return {-1, -1};
        return {x, y};
    }

    pair <int, int> random_pointf(vvi bimage, vc deltas, vvi filter)
    {
        int x, y;
        vc nz = nonzero(bimage);
        int cnt = nz.size();
        x = nz[0].first;
        y = nz[0].second;
        while (cnt >= 0 && !filter[x][y])
        {
            cnt--;
            x = nz[cnt].first;
            y = nz[cnt].second;
        }
        if (!filter[x][y]) return {-1, -1};
        return {x, y};
    }

    void get_deltas(vc &deltas2, vc &mxdeltas, vc &deltas, int d)
    {
        for (int x = -d; x <= d; x++)
        {
            for (int y = -d; y <= d; y++)
            {
                if (abs(x) <= d / 2 && abs(y) <= d / 2)
                {
                    deltas.push_back({x, y});
                    if (sqrt((double) (x * x + y * y)) - (d / 2) <= 0.5) 
                        mxdeltas.push_back({x, y});
                }
                if (sqrt((ld) (x * x + y * y)) - d <= 0.5) 
                    deltas2.push_back({x, y});
            }
        }
    }

    vc get_path(int x, int y, int tx, int ty, pair <int, int> shp)
    {
        int xn, yn;
        vc ans;
        ld d, prevd = 1e9;
        vvi vis(shp.first, vi(shp.second, 0));
        while (1)
        {
            for (pair <int, int> val : var)
            {
                xn = x + val.first, yn = y + val.second;
                if (!in_image(xn, yn, shp) || vis[xn][yn]) continue;
                if (xn == tx && yn == ty)
                {
                    ans.push_back({tx, ty});
                    return ans;
                }
                d = sqrt((ld)((x - tx) * (x - tx) + (y - ty) * (y - ty)));
                if (d < prevd)
                {
                    x = xn; y = yn;
                    vis[x][y] = 1;
                    prevd = d;
                    ans.push_back({x, y});
                    break;
                }
            }
        }
    }

    vc get_trajectory(vvi bimage, int d, int sx, int sy)
    {
        vc deltas2, mxdeltas, deltas;
        get_deltas(deltas2, mxdeltas, deltas, d);

        vc ans;

        vvi filter = bimage;

        int x, y, xn, yn, dtx, dty, xp, yp, it, xnn, ynn;
        bool change;
        
        pair <int, int> res = random_point(bimage, mxdeltas);
        x = res.first;
        y = res.second;
        xp = 0;
        yp = 0;
        it = 0;

        pair <int, int> shp = {bimage.size(), bimage[0].size()};

        while (neqvii(bimage, 0))
        {
            change = 0;
            for (pair <int, int> dt : deltas2)
            {
                dtx = dt.first; dty = dt.second;
                xn = x + dtx; yn = y + dty;
                if (in_image(xn, yn, shp) && bimage[xn][yn] && check_near(xn, yn, mxdeltas, bimage))
                {
                    x = xn; y = yn;
                    ans.push_back({x + sx, y + sy});
                    change = 1;
                    break;
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
                ans.push_back({x + sx, y + sy});
            }

            if (it)
            {
                for (pair <int, int> xyn : get_path(xp, yp, x, y, shp))
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
                    xn = x + dt.first; yn = x + dt.second;
                    if (in_image(xn, yn, shp)) bimage[xn][yn] = 0;
                }
            }
            xp = x;
            yp = y;
            it++;
        }
        return ans;
    }

    bool check_poly(vc cords, ld a, ld b)
    {
        ld s = 0;
        for (pair <int, int> pnt : cords)
            s = max(s, abs(a * pnt.first + b - pnt.second));
        return s < 5;
    }

    ld mean(vi vec)
    {
        ld s = 0;
        for (int e : vec) s += e;
        return s / vec.size();
    }

    ld variance(vi lst, ld mn)
    {
        ld s = 0;
        for (int e : lst)
            s += (e - mn) * (e - mn);
        return s / lst.size();
    }

    ld variancexy(vi x, vi y, ld xm, ld ym)
    {
        ld s = 0;
        for (int i = 0; i < x.size(); i++)
            s += (x[i] - xm) * (y[i] - ym);
        return s / x.size();
    }

    pair <bool, pair <ld, ld>> get_line(vc pnts)
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

        cout << sx << ' ' << sy << ' ' << sxy << '\n';
        if (sxy == 0) return {0, {0, 0}};
        ld a = (sy - sx + sqrt((sy - sx) * (sy - sx) + 4 * sxy * sxy)) / (2 * sxy);
        ld b = ym - a * xm;
        return {1, {a, b}};
    }

    vc approximate(vc cords)
    {
        int i = 0; 
        vc ncords;
        vc now;
        ld a = 1, b = 0, an, bn, cn;
        while (i < cords.size())
        {
            now.push_back(cords[i]);
            if (now.size() < 2)
            {
                i++;
                continue;
            }
        }
    }

    vvi pointer2vector2d(int *pointer, size_t n, size_t m)
    {
        vvi ans(n, vi(m));
        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < m; y++) ans[x][y] = pointer[x * n + y];
        }
        return ans;
    }

    void pointer2vectorpair(int *pointer, size_t n, vc &ans)
    {
        for (int i = 0; i < n; i++) 
        {
            ans[i].first = pointer[i * 2];
            ans[i].second = pointer[i * 2 + 1]; 
        }
    }

    void test_approximation(int* pointer, size_t n, ld* a, ld* b)
    {
        vc cords(n);
        pointer2vectorpair(pointer, n, cords);
        pair <bool, pair <ld, ld>> res = get_line(cords);
        cout << "got line" << '\n';
        *a = res.second.first;
        *b = res.second.second;
    }
}   