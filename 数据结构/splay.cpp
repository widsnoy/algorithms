#include <bits/stdc++.h>
using namespace std;
using ll = long long;

#define rank abcdefg
const int mod = 998244353;
const int N = 1e5 + 5;

int tot, fa[N], tr[N][2], sz[N], cnt[N], val[N], rt;

void maintain(int x) {
    sz[x] = sz[tr[x][0]] + sz[tr[x][1]] + cnt[x];
}
int getdir(int x) {
    return tr[fa[x]][1] == x;
}
void clear(int x) {
    fa[x] = sz[x] = cnt[x] = tr[x][0] = tr[x][1] = val[x] = 0;
}
int create(int v) {
    ++tot;
    val[tot] = v;
    sz[tot] = cnt[tot] = 1;
    return tot;
}
void rotate(int x) {
    if (x == rt) return;
    int y = fa[x], z = fa[y], d = getdir(x);
    tr[y][d] = tr[x][d ^ 1];
    if (tr[x][d ^ 1]) fa[tr[x][d ^ 1]] = y;
    fa[y] = x;
    tr[x][d ^ 1] = y;
    fa[x] = z;
    if (z) tr[z][y == tr[z][1]] = x;
    maintain(y);
    maintain(x);
}
void splay(int x) {
    for (int f = fa[x]; f = fa[x], f; rotate(x)) {
        if (fa[f]) rotate(getdir(f) == getdir(x) ? f : x);
    }
    rt = x;
}
void insert(int v) {
    if (!rt) {
        rt = create(v);
        return;
    }
    int u = rt, f = 0;
    while (true) {
        if (val[u] == v) {
            cnt[u]++;
            maintain(u);
            maintain(f);
            splay(u);
            return;
        }
        f = u, u = tr[u][v > val[u]];
        if (u == 0) {
            int id;
            fa[id = create(v)] = f;
            tr[f][v > val[f]] = id;
            maintain(f);
            splay(id);
            return;
        }
    }
}

int rank(int v) {
    int rk = 0;
    int u = rt;
    while (u) {
        if (val[u] == v) {
            rk += sz[tr[u][0]];
            splay(u);
            return rk + 1;
        }
        if (v < val[u]) {
            u = tr[u][0];
        } else {
            rk += sz[tr[u][0]] + cnt[u];
            u = tr[u][1];
        }
    }
    return -1;
}

int kth(int x) {
    int u = rt;
    while (u) {
        if (sz[tr[u][0]] + cnt[u] >= x && sz[tr[u][0]] < x) return val[u];
        if (x <= sz[tr[u][0]]) {
            u = tr[u][0];
        } else {
            x -= sz[tr[u][0]] + cnt[u];
            u = tr[u][1];
        }
    }
    return u ? val[u] : -1;
}
int pre() {
    int u = tr[rt][0];
    if (!u) return val[rt];
    while (true) {
        if (tr[u][1] == 0) return splay(u), val[u];
        u = tr[u][1];
    }
    return 233;
}
int suf() {
    int u = tr[rt][1];
    if (!u) return val[rt];
    while (true) {
        if (tr[u][0] == 0) return splay(u), val[u];
        u = tr[u][0];
    }
    return 233;
}
void del(int v) {
    if (rank(v) == -1) return;
    if (cnt[rt] > 1) {
        cnt[rt]--;
        return;
    }
    if (!tr[rt][1] && !tr[rt][0]) {
        clear(rt), rt = 0;
    } else if (!tr[rt][0]) {
        int x = rt;
        rt = tr[x][1];
        fa[rt] = 0;
        clear(x);
    } else if (!tr[rt][1]) {
        int x = rt;
        rt = tr[x][0];
        fa[rt] = 0;
        clear(x);
    } else {
        int cur = rt, y = tr[cur][1];
        pre();
        tr[rt][1] = y;
        fa[y] = rt;
        clear(cur);
        maintain(rt);
    }
}

int main() {
    int n, opt, x;

    for (scanf("%d", &n); n; --n) {
        scanf("%d%d", &opt, &x);

        if (opt == 1)
            insert(x);
        else if (opt == 2)
            del(x);
        else if (opt == 3)
            printf("%d\n", rank(x));
        else if (opt == 4)
            printf("%d\n", kth(x));
        else if (opt == 5)
            insert(x), printf("%d\n", pre()), del(x);
        else
            insert(x), printf("%d\n", suf()), del(x);
    }

    return 0;
}