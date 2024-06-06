#include <math.h>
#include <vector>
#include <deque>

using namespace std;

typedef long double ld;
typedef vector <vector <vector <ld>>> vvvd;
typedef vector <vector <ld>> vvd;
typedef vector <vector <int>> vvi;
typedef vector <int> vi;
typedef vector <ld> vd;

struct point 
{
    ld x, y;
    point(ld _x = 0, ld _y = 0): x(_x), y(_y) {};

    ld len()
    {
        return sqrt(x * x + y * y);
    }
};

ld dist(point a, point b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

point operator + (const point &a, const point &b) 
{
    return point(a.x + b.x, a.y + b.y);
}

point operator - (const point &a, const point &b) 
{
    return point(a.x - b.x, a.y - b.y);
}

ld operator * (const point &a, const point &b) 
{ 
    return a.x * b.x + a.y * b.y;
}

point operator * (const point &a, const double &k)
{
    return point(a.x * k, a.y * k);
}

ld operator % (const point &a, const point &b) 
{  
    return a.x * b.y - a.y * b.x;
}

point operator / (const point &a, const double &k)
{
    return point(a.x / k, a.y / k);
}

bool operator == (const point &a, const point &b)
{
    return dist(a, b) < 1e-6;
}

ld get_angle(point a, point b)
{
    return atan2(a % b, a * b);
}

bool operator == (const vd &a, const vd &b)
{
    for (int i = 0; i < a.size(); i++)
    {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

bool operator != (const vd &a, const vd &b)
{
    return !(a == b);
}

bool operator == (const vvi &a, const vvi &b)
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

vd operator - (const vd &a, const vd &b)
{
    vd ans(a.size());
    for (int i = 0; i < a.size(); i++) ans[i] = a[i] - b[i];
    return ans;
}

vvd operator - (const vd &a, const vvd &b)
{
    vvd ans(b.size());
    for (int i = 0; i < b.size(); i++) ans[i] = a - b[i];
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

vd abs(vd x)
{
    vd ans(x.size());
    for (int i = 0; i < x.size(); i++) ans[i] = abs(x[i]);
    return ans;
}

ld sum(vd x)
{
    ld ans = 0;
    for (int i = 0; i < x.size(); i++) ans += x[i];
    return ans;
}

vd sum(vvd x)
{
    vd ans(x.size());
    for (int i = 0; i < x.size(); i++) ans[i] = sum(x[i]);
    return ans;
}

vvi inside_or(vector <vector <pair <int, int>>> &arr)
{
    vvi ans(arr.size(), vi(ans[0].size(), 0));
    for (int x = 0; x < arr.size(); x++)
    {
        for (int y = 0; y < arr[0].size(); y++) ans[x][y] = arr[x][y].first || arr[x][y].second;
    }
    return ans;
}

vector <pair <int, int>> var = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

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
        if (px == zrs) nimg[i] = itsa[argmin(abs(sum(px - itsa)))];
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

vector <pair <int, int>> nonzero(vvi vec)
{
    vector <pair <int, int>> ans;
    for (int x = 0; x < vec.size(); x++)
    {
        for (int y = 0; y < vec[0].size(); y++)
        {
            if (vec[x][y]) ans.push_back({x, y});
        }
    }
    return ans;
}

bool check_near(int x, int y, vector <pair <int, int>> deltas, vvi bimage)
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

bool check_near(int x, int y, vector <pair <int, int>> deltas, vvi bimage, vvi f)
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
            if (!f[xn][yn]) return 0;
        }
    }
    return 1;
}

pair <int, int> random_point(vvi bimage, vector <pair <int, int>> deltas)
{
    int x, y;
    vector <pair <int, int>> nz = nonzero(bimage);
    int cnt = nz.size();
    x = nz[0].first;
    y = nz[0].second;
    while (cnt >= 0 && !check_near(x, y, deltas, bimage))
    {
        cnt--;
        x = nz[cnt].first;
        y = nz[cnt].second;
    }
    return {x, y};
}