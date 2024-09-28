#set page(
  paper: "us-letter",
  header: align(left)[
    _endless rain: widsnoy, WQhuanm, xu826281112_
  ],
  flipped: true
)
#show: rest => columns(2, rest)
#set heading(
  numbering: "1."
)
#set text(12pt)
#let style-number(number) = text(gray)[#number]
#show raw.where(block: true): it => block(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[#grid(columns: (1em, 1fr), align: (right, left), column-gutter: 0.7em, row-gutter: 0.6em, ..it.lines
  .enumerate()
  .map(((i, line)) => (style-number(i + 1), line))
  .flatten())]

#outline(
  title: [_widsnoy's *template*_],
  indent: auto
)
#colbreak()

= 数论
== 取模还原分数

== 原根
- 阶：$"ord"_m (a)$ 是最小的正整数 $n$ 使 $ a^n equiv 1 (mod m)$

- 原根：若 $g$ 满足 $(g,m)=1$ 且 $"ord"_m (g) eq phi(m)$ 则 $g$ 是 $m$ 的原根。若 $m$ 是质数，有 $g^i mod m, 0<i<m$ 的取值各不相同。

原根的应用：$m$ 是质数时，若求$a_k=sum_(i*j mod m eq k)f_i*g_j$ 可以通过原根转化为卷积形式(要求 $0$ 处无取值)。具体而言，$[1,m-1]$ 可以映射到 $g^([1,m-1])$，原式变为 $a_(g^k)=sum_(g^(i+j mod (m-1)) eq g^k)f_(g^i)*g_(g^j)$，令 $f_i eq f_(g^i)$ 则 $a_k=sum_((i+j) mod (m-1) eq k)f_i*g_j$

```cpp
int q[10005];
int getG(int n) {
    int i, j, t = 0;
    for (i = 2; (ll)(i * i) < n - 1; i++) {
        if ((n - 1) % i == 0) q[t++] = i, q[t++] = (n - 1) / i;
    }
    for (i = 2; ;i++) {
        for (j = 0; j < t;  j++) if (fpow(i, q[j], n) == 1) break;
        if (j == t) return i;
    }
    return -1;
}

vector<int> fpow(int kth) {
    if (kth == 0) return e;
    auto r = fpow(kth - 1);
    r = multiply(r, r);
    for (int i = p - 1; i < r.size(); i++) r[i % (p - 1)] = (r[i % (p - 1)] + r[i]) % mod;
    r.resize(p - 1);
    if (kk[kth] == '1') {
        r = multiply(r, e);
        for (int i = p - 1; i < r.size(); i++) r[i % (p - 1)] = (r[i % (p - 1)] + r[i]) % mod;
        r.resize(p - 1);
    }
    return r;
}
void MAIN() {
    g = getG(p);
    int tmp = 1;
    for (int i = 1; i < p; i++) {
        tmp = tmp * 1ll * g % p;
        mp[tmp] = i % (p - 1);
    }
    e.resize(p - 1);
    for (int i = 0; i < p - 1; i++) e[i] = 0;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {
            if (binom[i][j] == 0) continue;
            e[mp[binom[i][j]]]++;
        }
    }
}
```
== 解不定方程

== 中国剩余定理

== 卢卡斯定理

== exBSGS

== 二次剩余

== Miller-Rabin

== Pollard-rho

== 数论卷积

== 莫比乌斯反演 (两种形式)

== 除法分块 (上下取整)

== 杜教筛

== Min25 筛

== 区间筛


= 动态规划
== 缺1背包

= 图论
== 找环
```cpp
const int N = 5e5 + 5;
int n, m, col[N], pre[N], pre_edg[N];
vector<pii> G[N];
vector<vector<int>> resp, rese;
//point
void get_cyc(int u, int v) {
    if (!resp.empty()) return;
    vector<int> cyc;
    cyc.push_back(v);
    while (true) {
        v = pre[v];
        if (v == 0) break;
        cyc.push_back(v);
        if (v == u) break;
    }
    reverse(cyc.begin(), cyc.end());
    resp.push_back(cyc);
}
// edge
void get_cyc(int u, int v, int id) {
    if (!rese.empty()) return;
    vector<int> cyc;
    cyc.push_back(id);
    while (true) {
        if (pre[v] == 0) break;
        cyc.push_back(pre_edg[v]);
        v = pre[v];
        if (v == u) break;
    }
    reverse(cyc.begin(), cyc.end());
    rese.push_back(cyc);
}
void dfs(int u, int edg) {
    col[u] = 1;
    for (auto [v, id] : G[u]) if (id != edg) {
        if (col[v] == 1) {
            get_cyc(v, u);
            get_cyc(v, u, id);
        } else if (col[v] == 0) {
            pre[v] = u;
            pre_edg[v] = id;
            dfs(v, id);         
        }
    }
    col[u] = 2;
}
void MAIN() {
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        int u, v; cin >> u >> v;
        // G[u].push_back({v, i});
        // G[v].push_back({u, i});
    }
    for (int i = 1; i <= n; i++) if (!col[i]) dfs(i, -1);
}
```

== SPFA乱搞
```cpp
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

const int mod = 998244353;
const int N = 5e5 + 5;
const ll inf = 1e17;
int n, m, s, t, q[N], ql, qr;
int vis[N], fr[N];
ll dis[N];
vector<pii> G[N];
void MAIN() {
    cin >> n >> m >> s >> t;
    for (int i = 1; i <= m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        G[u].push_back({v, w});
    }
    for (int i = 0; i <= n; i++) dis[i] = inf;
    dis[s] = 0; q[qr] = s; vis[s] = 1;
    while (ql <= qr) {
        if (rng() % (qr - ql + 1) == 0) sort(q + ql, q + qr + 1, [](int x, int y) {
            return dis[x] < dis[y];
        });
        int u = q[ql++];
        vis[u] = 0;
        for (auto [v, w] : G[u]) {
            if (dis[u] + w < dis[v]) {
                dis[v] = dis[u] + w;
                fr[v] = u;
                if (!vis[v]) {
                    if (ql > 0) q[--ql] = v;
                    else q[++qr] = v;
                    vis[v] = 1;
                }
            }
        }
    }
    if (dis[t] == inf) {
        cout << "-1\n";
        return;
    }
    cout << dis[t] << ' ';
    vector<pii> stk;
    while (t != s) {
        stk.push_back({fr[t], t});
        t = fr[t];
    }
    reverse(stk.begin(), stk.end());
    cout << stk.size() << '\n';
    for (auto [u, v] : stk) cout << u << ' ' << v << '\n';   
}
```

== 差分约束

== 竞赛图

== 有向图强连通分量
=== Tarjan
```cpp
const int N = 5e5 + 5;
int n, m, dfc, dfn[N], low[N], stk[N], top, idx[N], in_stk[N], scc_cnt;
vector<int> G[N];

void tarjan(int u) {
    low[u] = dfn[u] = ++dfc;
    stk[++top] = u;
    in_stk[u] = 1;
    for (int v : G[u]) {
        if (!dfn[v]) {
            tarjan(v);
            low[u] = min(low[u], low[v]);
        } else if (in_stk[v]) low[u] = min(dfn[v], low[u]);
    }
    if (low[u] == dfn[u]) {
        int x;
        scc_cnt++;
        do {
            x = stk[top--];
            idx[x] = scc_cnt;
            in_stk[x] = 0;
        } while (x != u);
    }
}

void MAIN() {
    for (int i = 1; i <= n; i++) low[i] = dfn[i] = idx[i] = in_stk[i] = 0;
    dfc = scc_cnt = top = 0;
    cin >> n >> m;
    for (int i = 1; i <= n; i++) if (!dfn[i]) tarjan(i);
}
```
=== Kosaraju

== 强连通分量(incremental)

== 连通分量
=== 割点

=== 桥

=== 点双

=== 边双


== 二分图匹配
=== 匈牙利算法
=== KM

== 网络流
=== 网络最大流

=== 最小费用最大流
==== spfa
==== zkw
=== 上下界网络流

== 2-SAT
=== 搜索 (最小字典序)
=== tarjan

== 生成树
=== Prime
=== Kruskal
=== 次小生成树
=== 生成树计数

== 三元环

== 四元环

== 欧拉路

== 曼哈顿路

== 建图优化
=== 前后缀优化

=== 线段树优化

= 树论
== prufer

== 圆方树
=== 广义

=== 仙人掌

== 最近公共祖先

== 树分治
=== 点分治
=== 点分树

== 链分治
=== 重链分治

=== 长链分治

== dsu on tree

= 数学
== 组合恒等式
== min-max容斥
== 序列容斥
== 二项式反演
== 斯特林数
== 高维前缀和
== 线性基
== 行列式
== 高斯消元


= 多项式
== 快速数论变换

== 快速傅里叶变换

== 任意模数NTT

== 自然数幂和

== 快速沃尔什变换

== 子集卷积

= 数据结构

== 线段树
=== 李超树 (最大，次大，第三大)
=== 合并分裂
=== 线段树二分
=== 兔队线段树
== 平衡树
=== 文艺平衡树
== 历史版本信息线段树

== 树状数组二分

== 二维树状数组

== ODT

== KDT

== 手写堆

= 字符串
== KMP
== exKMP
== SA
== AC自动机
== 马拉车

= 杂项
== gcd, xor, or 分块
== 超级钢琴
== 平方计数
== FFT 字符串匹配
== 循环矩阵乘法
== 线性逆元
== 快快速幂
== 数学题基本预处理
== fastio
== 高精度

= 配置相关
== 对拍
== vscode 配置