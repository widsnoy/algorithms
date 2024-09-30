#set page(
  paper: "a4",
  header: align(left)[
    _endless rain: widsnoy, WQhuanm, xu826281112_
  ]
)
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
#pagebreak()

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

给出a,b,c,x1,x2,y1,y2，求满足 ax+by+c=0，且x∈[x1,x2],y∈[y1,y2]的整数解有多少对？
输入格式

第一行包含7个整数，a,b,c,x1,x2,y1,y2，整数间用空格隔开。

a,b,c,x1,x2,y1,y2的绝对值不超过$10^8$。

```cpp
#define y1 miku

ll a, b, c, x1, x2, y1, y2;
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b) {
        ll d = exgcd(b, a % b, y, x);
        return y -= a / b * x, d;
    } return x = 1, y = 0, a;
}

pll get_up(ll a, ll b, ll x1, ll x2) {
    //x2>=ax+b>=x1
    if (a == 0) return (b >= x1 && b <= x2) ? (pll){-1e18, 1e18} : (pll){1, 0};
    ll L, R;
    ll l = (x1 - b) / a - 3;
    for (L = l; L * a + b < x1; L++);
    ll r = (x2 - b) / a + 3;
    for (R = r; R * a + b > x2; R--);
    return {L, R}; 
}
pll get_dn(ll a, ll b, ll x1, ll x2) {
    //x2>=ax+b>=x1
    if (a == 0) return (b >= x1 && b <= x2) ? (pll){-1e18, 1e18} : (pll){1, 0};
    ll L, R;
    ll l = (x2 - b) / a - 3;
    for (L = l; L * a + b > x2; L++);
    ll r = (x1 - b) / a + 3;
    for (R = r; R * a + b < x1; R--);
    return {L, R}; 
}

void MAIN() {
    cin >> a >> b >> c >> x1 >> x2 >> y1 >> y2;
    if (a == 0 && b == 0) return cout << (c == 0) * (y2 - y1 + 1) * (x2 - x1 + 1) << '\n', void();
    ll x, y, d = exgcd(a, b, x, y);
    c = -c;
    if (c % d != 0) return cout << "0\n", void();
    x *= c / d, y *= c / d;
    ll sx = b / d, sy = -a / d;
    //x + k * sx  y + k * sy
    // 0<= 3 - k <= 4 [-1,3] [0,4]
    auto A = (sx > 0 ? get_up(sx, x, x1, x2) : get_dn(sx, x, x1, x2));
    auto B = (sy > 0 ? get_up(sy, y, y1, y2) : get_dn(sy, y, y1, y2));
    A.fi = max(A.fi, B.fi), A.se = min(A.se, B.se);
    cout << max(0ll, A.se - A.fi + 1) << '\n';
}
```

== 中国剩余定理
考虑合并两个同余方程

$
cases(
    x equiv a_1 (mod m_1),
    x equiv a_2 (mod m_2)
)
$

改写为不定方程形式

$
cases(
    x+m_1y=a_1,
    x+m_2y=a_2
)
$

取解集公共部分 $x=a_1-m_1 y_1=a_2- m_2 y_2$, 若$gcd(m_1, m_2)| (a_1-a_2)$ 有解，可以得到$x=k"lcm"(m_1,m_2)+a_2- m_2 y_2$ 化为同余方程的形式：$x equiv a_2- m_2y_2 (mod "lcm"(m_1,m_2))$
```cpp
ll n, m, a;
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b != 0) {
        ll g = exgcd(b, a % b, y, x);
        return y -= a / b * x, g;
    } return x = 1, y = 0, a;
}
ll getinv(ll a, ll mod) {
    ll x, y;
    exgcd(a, mod, x, y);
    x = (x % mod + mod) % mod;
    return x;
}
int get(ll x) {
    return x < 0 ? -1 : 1;
}
ll mul(ll a, ll b, ll mod) {
    ll res = 0;
    if (a == 0 || b == 0) return 0;
    ll f = get(a) * get(b);
    a = abs(a), b = abs(b);
    for (; b; b >>= 1, a = (a + a) % mod) if (b & 1) res = (res + a) % mod;
    res *= f;
    if (res < 0) res += mod;
    return res;
}
// m 互质 
// int main() {
//     cin >> n;
//     ll phi = 1;
//     for (int i = 1; i <= n; i++) {
//         cin >> m[i] >> a[i];
//         phi *= m[i];
//     }
//     ll ans = 0;
//     for (int i = 1; i <= n; i++) {
//         ll p = phi / m[i], q = getinv(p, m[i]);
//         ans += mul(p, mul(q, a[i], phi), phi);
//         ans %= phi;
//     }
//     cout << ans << '\n';
// }
int main() {
    cin >> n;
    cin >> m >> a;
    for (int i = 2; i <= n; i++) {
        ll nm, na;
        cin >> nm >> na;
        ll x, y;
        ll g = exgcd(m, -nm, x, y), d = (na - a) / g, md = abs(nm / g);
        x = mul(x, d, md);
        ll lc = abs(m / g);
        lc *= nm;
        a = (a + mul(m, x, lc)) % lc;
        m = lc;
    }
    cout << a << '\n';
}
```

== 卢卡斯定理
- p 为质数
$
binom(n, m) mod p=binom(floor(n/p), floor(m/p))binom(n mod p, m mod p) mod p
$
- p 不为质数
其中 calc(n, x, p) 计算 $n!/x^y mod p$ 的结果，其中 $y$ 是 $n!$ 含有 $x$ 的个数

如果 $p$ 是质数，利用 Wilson 定理 $(p-1)! equiv -1 (mod p)$ 可以$O(log P)$ 的计算 calc。其他情况可以通过预处理 $n!/(n"以内所有"p"倍数的乘积")$ 达到同样的效果。

```cpp
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b) {
        ll d = exgcd(b, a % b, y, x);
        return y -= a / b * x, d;
    } else return x = 1, y = 0, a;
}
int getinv(ll v, ll mod) {
    ll x, y;
    exgcd(v, mod, x, y);
    return (x % mod + mod) % mod;
}
ll fpow(ll a, ll b, ll p) {
    ll res = 1;
    for (; b; b >>= 1, a = a * 1ll * a % p) if (b & 1) res = res * 1ll * a % p;
    return res;
}
ll calc(ll n, ll x, ll p) {
    if (n == 0) return 1;
    ll s = 1;
    for (ll i = 1; i <= p; i++) if (i % x) s = s * i % p; 
    s = fpow(s, n / p, p);
    for (ll i = n / p * p + 1; i <= n; i++) if (i % x) s = i % p * s % p; 
    return calc(n / x, x, p) * 1ll * s % p;
}
int get(ll x) {
    return x < 0 ? -1 : 1;
}
ll mul(ll a, ll b, ll mod) {
    ll res = 0;
    if (a == 0 || b == 0) return 0;
    ll f = get(a) * get(b);
    a = abs(a), b = abs(b);
    for (; b; b >>= 1, a = (a + a) % mod) if (b & 1) res = (res + a) % mod;
    res *= f;
    if (res < 0) res += mod;
    return res;
}
ll sublucas(ll n, ll m, ll x, ll p) {
    ll cnt = 0;
    for (ll i = n; i; ) cnt += (i = i / x);
    for (ll i = m; i; ) cnt -= (i = i / x);
    for (ll i = n - m; i; ) cnt -= (i = i / x);
    return fpow(x, cnt, p) * calc(n, x, p) % p * getinv(calc(m, x, p), p) % p * getinv(calc(n - m, x, p), p) % p;
}
ll lucas(ll n, ll m, ll p) {
    int cnt = 0;
    ll a[21], mo[21];
    for (ll i = 2; i * i <= p; i++) if (p % i == 0) {
        mo[++cnt] = 1;
        while (p % i == 0) mo[cnt] *= i, p /= i;
        a[cnt] = sublucas(n, m, i, mo[cnt]);
    }
    if (p != 1) mo[++cnt] = p, a[cnt] = sublucas(n, m, p, mo[cnt]);
    ll phi = 1;
    for (int i = 1; i <= cnt; i++) phi *= mo[i];
    ll ans = 0;
    for (int i = 1; i <= cnt; i++) {
        ll p = phi / mo[i], q = getinv(p, mo[i]);
        ans += mul(p, mul(q, a[i], phi), phi);
        ans %= phi;
    }
    return ans;
}
```
== BSGS
求解 $a^x equiv n (mod p)$, $a, p$ 不一定互质
```cpp
int BSGS(int a, int b, int p) {
	unordered_map<int, int> x;
	int m = sqrt(p + 0.5);
	int v = ni(fpow(a, m), p);
	int e = 1; x[1] = 0;
	for(int i = 1; i < m; i++) {
        e = e * 1ll * a % p;
        if(!x[e]) x[e] = i;
	}
	for(int i = 0; i <= m; i++) {
		if(x[b]) return i * m + x[b];
		b = b * 1ll * v % p;
	}
	return -1;
}
int exBSGS(int a, int n, int p) {
    int d, q = 0, sum = 1;
    a %= p, n %= p;
    if(a == 1 || n == 1) return 0;
    while((d = gcd(a, p)) != 1) {
        if(n % d) return -1; 
        q++; n /= d; p /= d;
        sum = (sum * 1ll * a / d) % p;
        if(sum == n) return q;
    }
    int v = ni(sum, p);
    n = n * 1ll * v % p;
    int ans = BSGS(a, n, p);
    if(ans == -1) return -1;
    return ans + q;
}
```
== 二次剩余（待补）

== Miller-Rabin（待补）

== Pollard-rho（待补）

== 数论函数
+ $phi(n)=n product (1-1/p)$

+ $mu(n)=cases(
    1\,n=1,
    (-1)^"质因子个数"\, n "无平方因子",
    0\, n "有平方因子" 
)$

+ $mu * id eq phi $, $mu * 1 = epsilon$, $phi * 1 = id$

-  有一个表格，$a_(i,j)=gcd(i,j)$, 支持某一列一行乘一个数，查询整个表格的和。  

因为 $gcd(n,m)=sum_(i divides n and i divides m)phi(i)$，对每个 $phi(i)$ 维护一个大小为 $floor(n/i)$ 的表格，初始值全是 $phi(i)$, $(x,y)$ 对应 $(x*i,y*i)$。对大表格的修改可以转化为对小表格的修改，只需要对每行每列维护一个懒标记就行。

== 莫比乌斯反演

1. 若 $f(n) eq sum_(d divides n) g(d)$, 则 $g(n) eq sum_(d divides n)mu(n/d)f(d)$  
$
sum_(d divides n)mu(n/d)f(d)&=sum_(d divides n)mu(n/d)sum_(k divides d)g(k)\
&=sum_(k divides n)g(k)sum_(d divides n/k)mu(d)\
&=sum_(k divides n)g(k)\[n/k eq 1\] = g(n)
$
2. 若 $f(n) eq sum_(n divides d) g(d)$, 则 $g(n) eq sum_(n divides d)mu(d/n)f(d)$ 

3. $d(n m)=sum_(i divides n)sum_(j divides m)[gcd(i,j)=1]$ 

常见的一些推式子套路：
+ 证明是否积性函数，只需要观察是否满足 $f(p^i)f(q^j)=f(p^i q^j)$ 即可，用线性筛积性函数也是同理。

+ 形如 $sum_(d divides n) mu(d)sum_(k divides n/d)phi(k)floor(n/(d k))$ 的式子，这时候令 $T=d k$，枚举 $T$ 就能得到 $d,k$ 一个卷积的形式。如果是底数和指数，这时候不能线性筛，但是可以调和级数暴力算函数值。

== 整除分块

+ 下取整
```cpp
for (int i = 1, j; i <= min(n, m); i = j + 1) {
    j = min(n / (n / i), m / (m / i));
    // n / {i,...,j} = n / i 
}
```
+ 上取整
$ceil(n/i)=floor((n+i-1)/i)=floor((n-1)/i)+1$

== 区间筛

- 求解一个区间内的素数

如果是合数那么一定不大于 $sqrt(x)$ 的约数，使用这个范围内的数埃氏筛即可。

== 杜教筛



== Min25 筛


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

$"edge"[3]$ 保存了每条边的两个点在同一个强连通分量的时间。调用的时候右端点时间要大一位，因为可能有些边到最后也不能在一个强连通分量中。

```cpp
int n, m, Q, s[N];
vector<array<int, 4>> edge;
vector<int> G[N];
struct DSU {
    int fa[N], dep[N], top;
    pii stk[N];
    void init(int n) {
        top = 0;
        iota(fa, fa + n + 1, 0);
        fill(dep, dep + n + 1, 1);
    }
    int find(int u) {
        return u == fa[u] ? u : find(fa[u]);
    }
    void merge(int u, int v) {
        u = find(u), v = find(v);
        if (u == v) return;
        if (dep[u] > dep[v]) swap(u, v);
        stk[++top] = {u, (dep[u] == dep[v] ? v : -1)};
        fa[u] = v;
        dep[v] += (dep[u] == dep[v]);
    }
    void rev(int tim) {
        while (tim < top) {
            auto [u, v] = stk[top--];
            fa[u] = u;
            if (v != -1) dep[v]--;
        }
    }
} D;
int stk[N], top, dfc, dfn[N], low[N], in_stk[N];
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
        do {
            x = stk[top--];
            D.merge(x, u);
            in_stk[x] = 0;
        } while (x != u);
    }
}

void solve(int l, int r, int a, int b) {
    if (l == r) {
        for (int i = a; i <= b; i++) edge[i][3] = l;
        return;
    }
    int mid = (l + r) >> 1;
    vector<int> node;
    for (int i = a; i <= b; i++) if (edge[i][0] <= mid) {
        int u = D.find(edge[i][1]), v = D.find(edge[i][2]);
        if (u != v) node.push_back(u), node.push_back(v), G[u].push_back(v);
    }
    int otp = D.top;
    for (int x : node) if (!dfn[x]) tarjan(x);
    vector<array<int, 4>> e1, e2;
    for (int i = a; i <= b; i++) {
        int u = D.find(edge[i][1]), v = D.find(edge[i][2]);
        if (edge[i][0] > mid || u != v) e2.push_back(edge[i]);
        else e1.push_back(edge[i]);
    }
    int s1 = e1.size(), s2 = e2.size();
    for (int i = a; i < a + s1; i++) edge[i] = e1[i - a];
    for (int i = a + s1; i <= b; i++) edge[i] = e2[i - a - s1];
    dfc = 0;
    for (int x : node) dfn[x] = low[x] = 0, vector<int>().swap(G[x]);
    vector<int>().swap(node);
    vector<array<int, 4>>().swap(e1);
    vector<array<int, 4>>().swap(e2);
    solve(mid + 1, r, a + s1, b);
    D.rev(otp);
    solve(l, mid, a, a + s1 - 1);
}
```
== 连通分量
=== 割点和桥
```cpp
int dfn[N], low[N], dfs_clock;
bool iscut[N], vis[N];
void dfs(int u, int fa) {
    dfn[u] = low[u] = ++dfs_clock;
    vis[u] = 1;
    int child = 0;
    for (int v : e[u]) {
        if (v == fa) continue;
        if (!dfn[v]) {
            dfs(v, u);
            low[u] = min(low[u], low[v]);
            child++;
            if (low[v] >= dfn[u]) iscut[u] = 1;
        } else if (dfn[u] > dfn[v] && v != fa) low[u] = min(low[u], dfn[v]);
        if (fa == 0 && child == 1) iscut[u] = 0;
    }
}
```
=== 点双
```cpp
#include <cstdio>
#include <vector>
using namespace std;
const int N = 5e5 + 5, M = 2e6 + 5;
int n, m;

struct edge {
  int to, nt;
} e[M << 1];

int hd[N], tot = 1;

void add(int u, int v) { e[++tot] = (edge){v, hd[u]}, hd[u] = tot; }

void uadd(int u, int v) { add(u, v), add(v, u); }

int ans;
int dfn[N], low[N], bcc_cnt;
int sta[N], top, cnt;
bool cut[N];
vector<int> dcc[N];
int root;

void tarjan(int u) {
  dfn[u] = low[u] = ++bcc_cnt, sta[++top] = u;
  if (u == root && hd[u] == 0) {
    dcc[++cnt].push_back(u);
    return;
  }
  int f = 0;
  for (int i = hd[u]; i; i = e[i].nt) {
    int v = e[i].to;
    if (!dfn[v]) {
      tarjan(v);
      low[u] = min(low[u], low[v]);
      if (low[v] >= dfn[u]) {
        if (++f > 1 || u != root) cut[u] = true;
        cnt++;
        do dcc[cnt].push_back(sta[top--]);
        while (sta[top + 1] != v);
        dcc[cnt].push_back(u);
      }
    } else
      low[u] = min(low[u], dfn[v]);
  }
}

int main() {
  scanf("%d%d", &n, &m);
  int u, v;
  for (int i = 1; i <= m; i++) {
    scanf("%d%d", &u, &v);
    if (u != v) uadd(u, v);
  }
  for (int i = 1; i <= n; i++)
    if (!dfn[i]) root = i, tarjan(i);
  printf("%d\n", cnt);
  for (int i = 1; i <= cnt; i++) {
    printf("%llu ", dcc[i].size());
    for (int j = 0; j < dcc[i].size(); j++) printf("%d ", dcc[i][j]);
    printf("\n");
  }
  return 0;
}
```

=== 边双
```cpp
#include <algorithm>
#include <cstdio>
#include <vector>

using namespace std;
const int N = 5e5 + 5, M = 2e6 + 5;
int n, m, ans;
int tot = 1, hd[N];

struct edge {
  int to, nt;
} e[M << 1];

void add(int u, int v) { e[++tot].to = v, e[tot].nt = hd[u], hd[u] = tot; }

void uadd(int u, int v) { add(u, v), add(v, u); }

bool bz[M << 1];
int bcc_cnt, dfn[N], low[N], vis_bcc[N];
vector<vector<int>> bcc;

void tarjan(int x, int in) {
  dfn[x] = low[x] = ++bcc_cnt;
  for (int i = hd[x]; i; i = e[i].nt) {
    int v = e[i].to;
    if (dfn[v] == 0) {
      tarjan(v, i);
      if (dfn[x] < low[v]) bz[i] = bz[i ^ 1] = 1;
      low[x] = min(low[x], low[v]);
    } else if (i != (in ^ 1))
      low[x] = min(low[x], dfn[v]);
  }
}

void dfs(int x, int id) {
  vis_bcc[x] = id, bcc[id - 1].push_back(x);
  for (int i = hd[x]; i; i = e[i].nt) {
    int v = e[i].to;
    if (vis_bcc[v] || bz[i]) continue;
    dfs(v, id);
  }
}

int main() {
  scanf("%d%d", &n, &m);
  int u, v;
  for (int i = 1; i <= m; i++) {
    scanf("%d%d", &u, &v);
    if (u == v) continue;
    uadd(u, v);
  }
  for (int i = 1; i <= n; i++)
    if (dfn[i] == 0) tarjan(i, 0);
  for (int i = 1; i <= n; i++)
    if (vis_bcc[i] == 0) {
      bcc.push_back(vector<int>());
      dfs(i, ++ans);
    }
  printf("%d\n", ans);
  for (int i = 0; i < ans; i++) {
    printf("%llu", bcc[i].size());
    for (int j = 0; j < bcc[i].size(); j++) printf(" %d", bcc[i][j]);
    printf("\n");
  }
  return 0;
}
```

== 二分图匹配

=== 匈牙利算法


=== KM

== 网络流
=== 网络最大流
```cpp
int head[N], cur[N], ecnt, d[N];
struct Edge {
    int nxt, v, flow, cap;
}e[];
void add_edge(int u, int v, int flow, int cap) {
    e[ecnt] = {head[u], v, flow, cap}; head[u] = ecnt++;
    e[ecnt] = {head[v], u, flow, 0}; head[v] = ecnt++;
}
bool bfs() {
    memset(vis, 0, sizeof vis);
    std::queue<int> q;
    q.push(s);
    vis[s] = 1;
    d[s] = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int i = head[u]; i != -1; i = e[i].nxt) {
            int v = e[i].v;
            if (vis[v] || e[i].flow >= e[i].cap) continue;
            d[v] = d[u] + 1;
            vis[v] = 1;
            q.push(v);
        }
    }
    return vis[t];
}
int dfs(int u, int a) {
    if (u == t || !a) return a;
    int flow = 0, f;
    for (int& i = cur[u]; i != -1; i = e[i].nxt) {
        int v = e[i].v;
        if (d[u] + 1 == d[v] && (f = dfs(v, std::min(a, e[i].cap - e[i].flow))) > 0) {
            e[i].flow += f;
            e[i ^ 1].flow -= f;
            flow += f;
            a -= f;
            if (!a) break;
        }
    }
    return flow;
}

```
=== 最小费用最大流
```cpp
const int inf = 1e9;
int head[N], cur[N], ecnt, dis[N], s, t, n, m, mincost;
bool vis[N];
struct Edge {
    int nxt, v, flow, cap, w;
}e[100002];
void add_edge(int u, int v, int flow, int cap, int w) {
    e[ecnt] = {head[u], v, flow, cap, w}; head[u] = ecnt++;
    e[ecnt] = {head[v], u, flow, 0, -w}; head[v] = ecnt++;
}
bool spfa(int s, int t) {
    std::fill(vis + s, vis + t + 1, 0);
    std::fill(dis + s, dis + t + 1, inf);
    std::queue<int> q;
    q.push(s);
    dis[s] = 0;
    vis[s] = 1;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for (int i = head[u]; i != -1; i = e[i].nxt) {
            int v = e[i].v;
            if (e[i].flow < e[i].cap && dis[u] + e[i].w < dis[v]) {
                dis[v] = dis[u] + e[i].w;
                if (!vis[v]) vis[v] = 1, q.push(v);
            }
        }
    }
    return dis[t] != inf;
}
int dfs(int u, int a) {
    if (vis[u]) return 0;
    if (u == t || !a) return a;
    vis[u] = 1;
    int flow = 0, f;
    for (int& i = cur[u]; i != -1; i = e[i].nxt) {
        int v = e[i].v;
        if (dis[u] + e[i].w == dis[v] && (f = dfs(v, std::min(a, e[i].cap - e[i].flow))) > 0) {
            e[i].flow += f;
            e[i ^ 1].flow -= f;
            flow += f;
            mincost += e[i].w * f;
            a -= f;
            if (!a) break;
        }
    }
    vis[u] = 0;
    return flow;
}
```
=== 上下界网络流（待学）

== 2-SAT

$2 * u$ 代表不选择，$2*u+1$ 代表选择。
=== 搜索 (最小字典序)
```cpp
vector<int> G[N * 2];
bool mark[N * 2];
int stk[N], top;
void build_G() {
    for (int i = 1; i <= n; i++) {
        int u, v;
        G[2 * u + 1].push_back(2 * v);
        G[2 * v + 1].push_back(2 * u);
    }
}
bool dfs(int u) {
    if (mark[u ^ 1]) return false;
    if (mark[u]) return true;
    mark[u] = 1;
    stk[++top] = u;
    for (int v : G[u]) {
        if (!dfs(v)) return false;
    }
    return true;
}
bool 2_sat() {
    for (int i = 1; i <= n; i++) {
        if (!mark[i * 2] && !mark[i * 2 + 1]) {
            top = 0;
            if (!dfs(2 * i)) {
                while (top) mark[stk[top--]] = 0;
                if (!dfs(2 * i + 1)) return 0;
            }
        }
    }
    return 1;
}
```
=== tarjan
如果对于一个*x* `sccno`比它的反状态 *x*∧1 的 `sccno` 要小，那么我们用 *x* 这个状态当做答案，否则用它的反状态当做答案。

== 生成树
=== Prime
```cpp
int n, m;
vector<pii> G[N];
ll dis[N];
int vis[N];
void MAIN() {
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        G[u].push_back({v, w});
        G[v].push_back({u, w});
    }
    for (int i = 1; i <= n; i++) dis[i] = 1e18, vis[i] = 0;
    priority_queue<pair<ll, int>> q;
    dis[1] = 0;
    q.push({-dis[1], 1});
    ll ans = 0;
    while (!q.empty()) {
        auto [val, u] = q.top(); q.pop();
        if (vis[u]) continue;
        vis[u] = 1;
        ans -= val;
        for (auto [v, w] : G[u]) if (dis[v] > w) {
            dis[v] = w;
            q.push({-w, v});
        }
    }
    cout << ans << '\n';
}
```
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
== 底数固定快速幂
== fastio
== 高精度

= 配置相关
== 对拍
== vscode 配置