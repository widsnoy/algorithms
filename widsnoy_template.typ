#set page(
  paper: "a4",
  header: align(left)[
    _hdu-t05: widsnoy, WQhuanm, xu826281112_
  ]
)
#set heading(
  numbering: "1."
)
#set text(
    size: 12pt,
    font: ("Linux libertine", "Noto Sans CJK SC"), lang: "zh", region: "cn")
#set page(numbering: "(i)")
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
#set page(numbering: "1")
#counter(page).update(1)
= 数论
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
        if ((na - a) % g) return -1;
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
int fpow(int a, int b, int p) {
    int res = 1;
    for (; b; b >>= 1, a = a * 1ll * a % p) if (b & 1) res = res * 1ll * a % p;
    return res;
}
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) return x = 1, y = 0, a;
    ll d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
int inv(int a, int p) {
    ll x, y; 
    ll g = exgcd(a, p, x, y);
    if (g != 1) return -1;
    return (x % p + p) % p;
}
int BSGS(int a, int b, int p) {
    if (p == 1) return 1;
    unordered_map<int, int> x;
    int m = sqrt(p + 0.5) + 1;
    int v = inv(fpow(a, m, p), p);
    int e = 1; 
    for(int i = 1; i <= m; i++) {
        e = e * 1ll * a % p;
        if(!x.count(e)) x[e] = i;
    }
    for(int i = 0; i <= m; i++) {
        if(x.count(b)) return i * m + x[b];
        b = b * 1ll * v % p;
    }
    return -1;
}
pii exBSGS(int a, int n, int p) {
    int d, q = 0, sum = 1;
    if (n == 1) return {0, gcd(a, p) == 1 ? BSGS(a, 1, p) : 0};
    a %= p, n %= p;
    while((d = gcd(a, p)) != 1) {
        if(n % d) return {-1, -1}; 
        q++; n /= d; p /= d;
        sum = (sum * 1ll * a / d) % p;
        if(sum == n) return {q, gcd(a, p) == 1 ? BSGS(a, 1, p) : 0};
    }
    int v = inv(sum, p);
    n = n * 1ll * v % p;
    int ans = BSGS(a, n, p);
    if(ans == -1) return {-1, -1};
    return {ans + q, BSGS(a, 1, p)};
}
```

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

== Min25 筛

能在 $O(n^(3/4)/log(n))$ 时间求出 $F(n)=sum_(i=1)^(n)f(i)$ 的值，要求积性函数能快速求出 $f(p^k)$ 处的点值。

- 定义 $R(i)$ 表示 $i$ 的最小质因子
$
G(n,j)=sum_(i=1)^n f(i)[i in "prime" or R(i) > P_j]
$
考虑递推
$
G(n,j)=cases(
    G(n,j-1) "IF" p_j times p_j > n,
    G(n,j-1)-f(p_j)(G(n/p_j,j-1)-sum_(i=1)^(j-1)f(p_i)) "IF" p_j times p_j <= n
)
$

根据整除分块，G 函数的第一维只用 $sqrt(n)$ 种取值，将其存在 $w[]$ 中，且用 $"id1"[]$ 和 $"id2"[]$ 分别存数字对应的下标位置。因为最后只需要知道 $G(x,"pcnt")$ 所以第二维可以滚掉。

- 定义 $S(n,j)=sum_(i=1)^n f(i)[R(i)>=p_j]$  

质数部分答案显然为 $G(n,"pcnt")-sum_(i=1)^(j-1)f(p_i)$, 合数部分考虑提出最小的质因子 $p^k$，得到 $S(n,j)$ 的递推式

$
S(n,j)=G(n,"pcnt")-sum_(i=1)^(j-1)f(p_i)+sum_(i=j)^"pcnt"sum_(k=1)^(p_i^(k+1)<=n)f(p^k)S(n/p^k,j+1)+f(p^(k+1))
$

递归边界是 $n=1 or p_j > n$, $S(n,j)=0$  

$sum_(i=1)^(n)f(i)=S(n,1)+f(1)$

```cpp
#include <cstdio>
#include <cmath>

typedef long long ll;
const int N = 4e6 + 5, MOD = 1e9 + 7;
const ll i6 = 166666668, i2 = 500000004;
ll n, id1[N], id2[N], su1[N], su2[N], p[N], sqr, w[N], g[N], h[N];
int cnt, m;
bool vis[N];

ll add(ll a, ll b) {a %= MOD, b %= MOD; return (a + b >= MOD) ? a + b - MOD : a + b;}
ll mul(ll a, ll b) {a %= MOD, b %= MOD; return a * b % MOD;}
ll dec(ll a, ll b) {a %= MOD, b %= MOD; return ((a - b) % MOD + MOD) % MOD;}

void init(int m) {
	for (ll i = 2; i <= m; i++) {
		if (!vis[i]) p[++cnt] = i, su1[cnt] = add(su1[cnt - 1], i), su2[cnt] = add(su2[cnt - 1], mul(i, i));
		for (int j = 1; j <= cnt && i * p[j] <= m; j++) {
			vis[p[j] * i] = 1;
			if (i % p[j] == 0) break;
		}
	}
}

ll S(ll x, int y) {
	if (p[y] > x || x <= 1) return 0;
	int k = (x <= sqr) ? id1[x] : id2[n / x];
	ll res = dec(dec(g[k], h[k]), dec(su2[y - 1], su1[y - 1]));
	for (int i = y; i <= cnt && p[i] * p[i] <= x; i++) {
		ll pow1 = p[i], pow2 = p[i] * p[i];
		for (int e = 1; pow2 <= x; pow1 = pow2, pow2 *= p[i], e++) {
			ll tmp = mul(mul(pow1, dec(pow1, 1)), S(x / pow1, i + 1));
			tmp = add(tmp, mul(pow2, dec(pow2, 1)));
			res = add(res, tmp);
		}
	}
	return res;
}

int main() {
    scanf("%lld", &n);
	sqr = sqrt(n + 0.5) + 1;
	init(sqr);
	for (ll l = 1, r; l <= n; l = r + 1) {
        r = n / (n / l);
		w[++m] = n / l;
		g[m] = mul(w[m] % MOD, (w[m] + 1) % MOD);
		g[m] = mul(g[m], (2 * w[m] + 1) % MOD);
		g[m] = mul(g[m], i6);
        g[m] = dec(g[m], 1);
		h[m] = mul(w[m] % MOD, (w[m] + 1) % MOD);;
	    h[m] = mul(h[m], i2);
		h[m] = dec(h[m], 1);
	    (w[m] <= sqr) ? id1[w[m]] = m : id2[r] = m;
	}
	for (int j = 1; j <= cnt; j++)
		for (int i = 1; i <= m && p[j] * p[j] <= w[i]; i++) {
			int k = (w[i] / p[j] <= sqr) ? id1[w[i] / p[j]] : id2[n / (w[i] / p[j])];
		    g[i] = dec(g[i], mul(mul(p[j], p[j]), dec(g[k], su2[j - 1])));
			h[i] = dec(h[i], mul(p[j], dec(h[k], su1[j - 1])));
		}
	//printf("%lld\n", g[1] - h[1]);
	printf("%lld\n", add(S(n, 1), 1));
	return 0;
}
```

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
== SPFA
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

== 连通分量
=== 有向图强连通分量
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
=== 强连通分量(incremental)

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

mch 记录的是右部点匹配的左部点

```cpp
int mch[maxn], vis[maxn];
std::vector<int> e[maxn];
bool dfs(const int u, const int tag) {
    for (auto v : e[u]) {
        if (vis[v] == tag) continue;
        vis[v] = tag;
        if (!mch[v] || dfs(mch[v], tag)) return mch[v] = u, 1;
    }
    return 0;
}
int main() {
    int ans = 0;
    for (int i = 1; i <= n; ++i) if (dfs(i, i)) ++ans;
}
```
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

== 2-SAT

$2 * u$ 代表不选择，$2*u+1$ 代表选择。
=== 搜索
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

== 圆方树
记得开两倍空间。
```cpp
void tarjan(int u) {
    stk[++top] = u;
    low[u] = dfn[u] = ++dfc;
    for (int v : G[u]) {
        if (!dfn[v]) {
            tarjan(v);
            low[u] = min(low[u], low[v]);
            if (low[v] == dfn[u]) {
                cnt++;
                for (int x = 0; x != v; --top) {
                    x = stk[top];
                    T[cnt].push_back(x);
                    T[x].push_back(cnt);
                    val[cnt]++;
                }
                T[cnt].push_back(u);
                T[u].push_back(cnt);
                val[cnt]++;
            }
        } else low[u] = min(low[u], dfn[v]);
    }
}
// 调用
cnt = n;
for (int i = 1; i <= n; i++) if (!dfn[i]) {
    tarjan(i);
    --top;
}
```
- 静态仙人掌最短路。边权设置为到点双顶点的最短距离。
```cpp
void tarjan(int u) {
    stk[++top] = u;
    dfn[u] = low[u] = ++dfc;
    for (auto [v, w] : G[u]) if (!dfn[v]) {
        dis[v] = dis[u] + w;
        tarjan(v);
        low[u] = min(low[u], low[v]);
        if (low[v] == dfn[u]) {
            ++cnt;
            val[cnt] = cyc[stk[top]] + dis[stk[top]] - dis[u];
            for (int x = 0; x != v; --top) {
                x = stk[top];
                //assert(val[cnt] >= (dis[x] - dis[u]));
                int w = min(dis[x] - dis[u], val[cnt] - (dis[x] - dis[u]));
                T[cnt].push_back({x, w});
                T[x].push_back({cnt, w});
            }
            T[cnt].push_back({u, 0});
            T[u].push_back({cnt, 0});
        }
    } else if (dfn[v] < dfn[u]) {
        cyc[u] = w;
        low[u] = min(low[u], dfn[v]);
    }
}

void dfs(int u, int fa) {
    faz[0][u] = fa;
    for (int k = 1; k < M; k++) faz[k][u] = faz[k - 1][faz[k - 1][u]];
    for (auto [v, w] : T[u]) if (v != fa) {
        dep[v] = dep[u] + 1;
        ff[v] = ff[u] + w;
        dfs(v, u);
    }
}
int dist(int u, int v) {
    int tu = u, tv = v;
    if (dep[u] < dep[v]) swap(u, v);
    int det = dep[u] - dep[v];
    for (int k = 0; k < M; k++) if ((det >> k) & 1) u = faz[k][u];
    int lca;
    if (u == v) lca = u;
    else {
        for (int k = M - 1; k >= 0; k--) if (faz[k][u] != faz[k][v]) {
            u = faz[k][u]; v = faz[k][v];
        }
        lca = faz[0][u];
    }
    if (lca <= n) return ff[tu] + ff[tv] - ff[lca] * 2;
    int tm = min(abs(dis[u] - dis[v]), val[lca] - abs(dis[u] - dis[v]));
    return ff[tu] - ff[u] + ff[tv] - ff[v] + tm;
}
```

- 圆方树上 dp

以单源最短路为例，原点记录该点出发是否返回的最长路，方点记录顶点出发经过环上所能走到的最长路。

```cpp
void dfs(int u, int fa) {
    for (int v : T[u]) if (v != fa) dfs(v, u);
    if (u <= n) {
        int mx = 0; 
        /*
        这里必须设为 0 而不是 -inf, 或者在平凡方点转移的时候要 max(dp[0], dp[1])
        hack: 4 4
        1 2
        2 3
        3 4
        4 2
        */
        for (int v : T[u]) if (v != fa) {
            dp[u][1] += dp[v][1];
            mx = max(mx, dp[v][0] - dp[v][1]);
            dp[u][0] += dp[v][1];
        }
       dp[u][0] += mx;
    } else {
        int sum = 1;
        dp[u][1] = 1;
        for (int v : T[u]) if (v != fa) {
            dp[u][1] += dp[v][1] + 1;
            dp[u][0] = max(dp[u][0], sum + dp[v][0]);
            sum += dp[v][1] + 1;
        }
        sum = 1;
        reverse(T[u].begin(), T[u].end());
        for (int v : T[u]) if (v != fa) {
            dp[u][0] = max(dp[u][0], sum + dp[v][0]);
            sum += dp[v][1] + 1;
        }
        if (val[u] == 2) dp[u][1] = 0;
    }
}
```

== 欧拉回路
- 有向图
```cpp
void dfs(int u) {
    for (int &i = hd[u]; i < G[u].size(); ) dfs(G[u][i++]);
    stk.push_back(u);
}
int check() {
    int mo = 0, le = 0, st = 1;
    for (int i = 1; i <= n; i++) {
        if (abs(in[i] - out[i]) > 1) return -1;
        if (in[i] > out[i]) le++;
        if (in[i] < out[i]) mo++, st = i;
    }
    if (mo > 1 || le > 1 || mo + le == 1) return -1;
    return st;
}

void MAIN() {
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        int u, v;
        cin >> u >> v;
        in[v]++; out[u]++;
        G[u].push_back(v);
    }
    for (int i = 1; i <= n; i++) sort(G[i].begin(), G[i].end());
    int tmp = check();
    if (tmp == -1) cout << "No\n";
    else {
        dfs(tmp);
        copy(stk.rbegin(), stk.rend(), ostream_iterator<int>(cout, " "));
        cout << '\n';
    }
}
```

- 无向图
```cpp
void dfs(int u) {
    for (int &i = hd[u]; i < G[u].size(); ) {
        while (i < G[u].size() && cnt[u][G[u][i]] == 0) ++i;
        if (i == G[u].size()) break;
        cnt[u][G[u][i]]--;
        cnt[G[u][i]][u]--;
        dfs(G[u][i++]);
    }
    stk.push_back(u);
}
int check() {
    int odd = 0, st = -1;
    for (int i = 1; i <= n; i++) {
        if (deg[i] == 0) continue;
        if (st == -1) st = i;
        if (deg[i] & 1) {
            ++odd;
            if (odd == 1) st = i;
        }
    }
    if (odd > 2) return -1;
    return st;
}

void MAIN() {
    n = 500;
    cin >> m;
    for (int i = 1; i <= m; i++) {
        int u, v;
        cin >> u >> v;
        ++deg[u]; ++deg[v];
        G[u].push_back(v);
        G[v].push_back(u);
        ++cnt[u][v];
        ++cnt[v][u];
    }
    for (int i = 1; i <= n; i++) sort(G[i].begin(), G[i].end());
    int tmp = check();
    if (tmp == -1) cout << "No\n";
    else {
        dfs(tmp);
        copy(stk.rbegin(), stk.rend(), ostream_iterator<int>(cout, "\n"));
    }
}
```

== 无向图三/四元环计数
- 三元环
```cpp
int vis[N];
vector<int> G[N];
ll main() {
    ll cnt = 0;
    for (int i = 0; i < m; i++) {
        if (deg[ed[i].fi] == deg[ed[i].se] && ed[i].fi > ed[i].se) swap(ed[i].fi, ed[i].se);
        if (deg[ed[i].fi] > deg[ed[i].se]) swap(ed[i].fi, ed[i].se);
        G[ed[i].fi].push_back(ed[i].se);
    }
    for (int u = 1; u <= n; u++) {
        for (int v : G[u]) vis[v] = 1;
        for (int v : G[u]) for (int w : G[v]) if (vis[w]) ++cnt;
        for (int v : G[u]) vis[v] = 0;
    }
    return cnt;
}
```

- 四元环

统计 $c?b->a<-d?c$ 的数目，因为最大度数点 $a$ 不同，所以不会算重。

```cpp
int n, m, deg[N], cnt[N];
bool bigger(int a, int b) {
    return deg[a] > deg[b] || (deg[a] == deg[b] && a > b);
}
void MAIN() {
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        int u, v;
        cin >> u >> v;
        ed.push_back({u, v});
        G[u].push_back(v);
        G[v].push_back(u);
        ++deg[u]; ++deg[v];
    }
    for (auto [u, v] : ed) {
        if (bigger(v, u)) swap(u, v);
        T[u].push_back(v);
    }
    ll ans = 0;
    for (int a = 1; a <= n; a++) {
        for (int b : T[a]) {
            for (int c : G[b]) {
                if (c == a || bigger(c, a)) continue;
                ans += cnt[c];
                ++cnt[c];
            }
        }
        for (int b : T[a]) for (int c : G[b]) cnt[c] = 0;
    }
    cout << ans << '\n';
}
```
== 虚树

需要保证 $"LCA"(0, u) = 0$

```cpp
int solve(vector<int>po) {
    sort(po.begin(), po.end(), [](int x, int y) {
        return dfn[x] < dfn[y];
    });
    int ans = 0;
    top = 0;
    stk[++top] = 0;
    for (int u : po) {
        int lca = LCA(u, stk[top]);
        if (lca == stk[top]) stk[++top] = u;
        else {
            for (int i = top; i >= 2 && dep[stk[i - 1]] >= dep[lca]; i--) {
              //  ans += ff[stk[i]] - ff[stk[i - 1]] - (vis[stk[i]] ? val[stk[i]]: 0); 
              //  cout << stk[i] << ' ' << stk[i - 1] << ' ' << ff[stk[i]] - ff[stk[i - 1]] - (vis[stk[i]] ? val[stk[i]]: 0) << '\n';
                add_edge(stk[i], stk[i - 1]);
                --top;
            }
            if (stk[top] != lca) {
              //  cout << lca << ' ' << stk[top] << ' ' << ff[stk[top]] - ff[lca] - (vis[stk[top]] ? val[stk[top]] : 0) << '\n';
              //  ans += ff[stk[top]] - ff[lca] - (vis[stk[top]] ? val[stk[top]] : 0);
                add_edge(stk[top], lca);
                stk[top] = lca;
            }
            stk[++top] = u;
        }
    }
    for (int i = 2; i < top; i++) {
      //  cout << stk[i + 1] << ' ' << stk[i] << ' ' << ff[stk[i + 1]] - ff[stk[i]] - (vis[stk[i + 1]] ? val[stk[i + 1]] : 0) << '\n';
       // ans += ff[stk[i + 1]] - ff[stk[i]] - (vis[stk[i + 1]] ? val[stk[i + 1]] : 0);
        add_edge(stk[i + 1], stk[i]);
    }
    //ans += (vis[stk[2]] ? 0 : val[stk[2]]);
    return ans;
}
```

== 最近公共祖先
```cpp
// 倍增
int faz[N][20], dep[N];
void dfs(int u, int fa) {
    faz[u][0] = fa;
    dep[u] = dep[fa] + 1;
    for (int i = 1; i < 20; i++) faz[u][i] = faz[faz[u][i - 1]][i - 1];
    for (int v : G[u]) if (v != fa) {
        dfs(v, u);
    }
}
int LCA(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);
    int d = dep[u] - dep[v];
    for (int i = 0; i < 20; i++) if ((d >> i) & 1) u = faz[u][i];
    if (v == u) return u;
    for (int i = 19; i >= 0; i--) if (faz[u][i] != faz[v][i]) 
        u = faz[u][i], v = faz[v][i];
    return faz[u][0];
}

//树剖
int dfc, dfn[N], rnk[N], siz[N], top[N], dep[N], son[N], faz[N];
void dfs1(int u, int fa) {
    dep[u] = dep[fa] + 1;
    siz[u] = 1;
    son[u] = -1;
    faz[u] = fa;
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs1(v, u);
        siz[u] += siz[v];
        if (son[u] == -1 || siz[son[u]] < siz[v]) son[u] = v;
    }
}
void dfs2(int u, int fa, int tp) {
    dfn[u] = ++dfc;
    rnk[dfc] = u;
    top[u] = tp;
    if (son[u] != -1) dfs2(son[u], u, tp);
    for (int v : G[u]) {
        if (v == fa || v == son[u]) continue;
        dfs2(v, u, v);
    }
}
int LCA(int u, int v) {
    while (top[u] != top[v]) {
        if (dep[top[u]] > dep[top[v]])
            u = faz[top[u]];
        else
            v = faz[top[v]];
    }
    return dep[u] > dep[v] ? v : u;
}

// O(1) query

int dfn[N], faz[N], dep[N], rnk[N], dfc, st[N][20];
void dfs(int u, int fa) {
    dfn[u] = ++dfc; faz[u] = fa; dep[u] = dep[fa] + 1; rnk[dfc] = u;
    for (auto [v, w] : G[u]) if (v != fa) dfs(v, u);
}
int LCA(int u, int v) {
    if (u == v) return u;
    if (dfn[u] > dfn[v]) swap(u, v);
    int l = dfn[u] + 1, r = dfn[v];
    int k = __lg(r - l + 1);
    return dep[st[l][k]] < dep[st[r - (1 << k) + 1][k]] ? faz[st[l][k]] : faz[st[r - (1 << k) + 1][k]];
}

int main() {
    dfs(1, 0);
    dep[0] = n + 1;
    for (int i = 1; i <= n; i++) st[i][0] = rnk[i];
    for (int j = 1; j < 20; j++) {
        for (int i = 1; i <= n; i++) {
            st[i][j] = dep[st[i][j - 1]] <= dep[st[min(n, i + (1 << (j - 1)))][j - 1]] ? st[i][j - 1] : st[min(n, i + (1 << (j - 1)))][j - 1];
        }
    }
}
```

= 数学
== 子集卷积
高维前缀和
```cpp
for (int k = 0; k < 20; k++) {
    for (int i = 0; i < (1 << 20); i++) if ((i >> k) & 1) {
        f[i] = f[i] + f[i ^ (1 << k)];
    }
}
```
高维后缀和
```cpp
for (int k = 0; k < 20; k++) {
    for (int i = 0; i < (1 << 20); i++) if ((i >> k) & 1) {
        f[i] = f[i] + f[i ^ (1 << k)];
    }
}
```
高维差分
```cpp
for (int k = 0; k < 20; k++) {
    for (int i = 0; i < (1 << 20); i++) if ((i >> k) & 1) {
        f[i] = f[i] - f[i ^ (1 << k)];
    }
}
```
== 线性基
```cpp
struct LinerBasis {
    int a[20], pos[20];
    void add(int v, int p) {
        for (int i = 19; i >= 0; i--) if ((v >> i) & 1) {
            if (a[i]) {
                if (p > pos[i]) {
                    swap(p, pos[i]);
                    swap(a[i], v);
                }
                v ^= a[i];
            } else {
                a[i] = v;
                pos[i] = p;
                return;
            }
        }
    }
} b[N];

LinerBasis operator + (LinerBasis a, LinerBasis b) {
    for (int i = 19; i >= 0; i--) {
        if (b.a[i]) a.add(b.a[i], b.pos[i]);
    }
    return a;
}
```
== 高斯消元
```cpp
namespace Gauss {
    bitset<258> a[256 + 256 + 5];
    int n;
    void push(const bitset<258>& x) {
        a[++n] = x;
    }
    bool solve(int m) {
        int k = 1;
        for (int i = 1; i <= m; i++) {
            if (k > n) break;
            for (int j = k + 1; j <= n; j++) if (a[j][i] > 0) {
                swap(a[k], a[j]);
                break;
            }
            if (a[k][i] == 0) break;
            for (int j = 1; j <= n; j++) if (j != k && a[j][i]) {
                a[j] ^= a[k];
            }
            ++k;
        }
        for (int i = k; i <= n; i++) if (a[i][m + 1]) return false;
        return true;
    }
}
```

= 多项式
== NTT

这个板子很慢

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef vector<int> poly;
const int mod = 998244353;
const int N = 4000000 + 5;

int rf[32][N];
int fpow(int a, int b) {
    int res = 1;
    for (; b; b >>= 1, a = a * 1ll * a % mod) if (b & 1)
        res = res * 1ll * a % mod;
    return res;
}
void init(int n) {
    assert(n < N);
    int lg = __lg(n);
    static vector<bool> bt(32, 0);
    if (bt[lg] == 1) return;
    bt[lg] = 1;
    for (int i = 0; i < n; i++) rf[lg][i] = (rf[lg][i >> 1] >> 1) + ((i & 1) ? (n >> 1) : 0);
}
void ntt(poly &x, int lim, int op) {
    int lg = __lg(lim), gn, g, tmp;;
    for (int i = 0; i < lim; i++) if (i < rf[lg][i]) swap(x[i], x[rf[lg][i]]);
    for (int len = 2; len <= lim; len <<= 1) {
        int k = (len >> 1);
        gn = fpow(3, (mod - 1) / len);
        for (int i = 0; i < lim; i += len) {
            g = 1;
            for (int j = 0; j < k; j++, g = gn * 1ll * g % mod) {
                tmp = x[i + j + k] * 1ll * g % mod;
                x[i + j + k] = (x[i + j] - tmp + mod) % mod;
                x[i + j] = (x[i + j] + tmp) % mod;
            }
        }
    }
    if (op == -1) {
        reverse(x.begin() + 1, x.begin() + lim);
        int inv = fpow(lim, mod - 2);
        for (int i = 0; i < lim; i++) x[i] = x[i] * 1ll * inv % mod;
    }
}
poly multiply(const poly &a, const poly &b) {
    assert(!a.empty() && !b.empty());
    int lim = 1;
    while (lim + 1 < int(a.size() + b.size())) lim <<= 1;
    init(lim);
    poly pa = a, pb = b;
    while (pa.size() < lim) pa.push_back(0);
    while (pb.size() < lim) pb.push_back(0);
    ntt(pa, lim, 1); ntt(pb, lim, 1);
    for (int i = 0; i < lim; i++) pa[i] = pa[i] * 1ll * pb[i] % mod;
    ntt(pa, lim, -1);
    while (int(pa.size()) + 1 > int(a.size() + b.size())) pa.pop_back();
    return pa;
}
poly prod_poly(const vector<poly>& vec) { // init vector, too slow
    int n = vec.size();
    auto calc = [&](const auto &self, int l, int r) -> poly {
        if (l == r) return vec[l];
        int mid = (l + r) >> 1;
        return multiply(self(self, l, mid), self(self, mid + 1, r));
    };
    return calc(calc, 0, n - 1);
}

// Semi-Online-Convolution
poly semi_online_convolution(const poly& g, int n, int op = 0) {
    assert(n == g.size());
    poly f(n, 0);
    f[0] = 1;
    auto CDQ = [&](const auto &self, int l, int r) -> void {
        if (l == r) {
            // exp
            if (op == 1 && l > 0) f[l] = f[l] * 1ll * fpow(l, mod - 2) % mod;
            return;
        }
        int mid = (l + r) >> 1;
        self(self, l, mid);
        poly a, b;
        for (int i = l; i <= mid; i++) a.push_back(f[i]);
        for (int i = 0; i <= r - l - 1; i++) b.push_back(g[i + 1]);
        a = multiply(a, b);
        for (int i = mid + 1; i <= r; i++) f[i] = (f[i] + a[i - l - 1]) % mod;
        self(self, mid + 1, r);
    };
    CDQ(CDQ, 0, n - 1);
    return f;
}

poly getinv(const poly &a) {
    assert(!a.empty());
    poly res = {fpow(a[0], mod - 2)}, na = {a[0]};
    int lim = 1;
    while (lim < int(a.size())) lim <<= 1;
    for (int len = 2; len <= lim; len <<= 1) {
        while (na.size() < len) {
            int tmp = na.size();
            if (tmp < a.size()) na.push_back(a[tmp]);
            else na.push_back(0);
        }
        auto tmp = multiply(na, res);
        for (auto &x : tmp) x = (x > 0 ? mod - x : x);
        tmp[0] = ((tmp[0] + 2) >= mod) && (tmp[0] -= mod);
        tmp = multiply(res, tmp);
        while (tmp.size() > len) tmp.pop_back();
        res = tmp;
    }
    while (res.size() > a.size()) res.pop_back();
    return res;
}
poly exp(const poly &g) {
    int n = g.size();
    poly b(n, 0);
    for (int i = 1; i < n; i++) b[i] = i * 1ll * g[i] % mod;
    return semi_online_convolution(b, n, 1);
}
poly ln(const poly &A) {
    int n = A.size();
    auto C = getinv(A);
    poly A1(n, 0);
    for (int i = 0; i < n - 1; i++) A1[i] = (i + 1) * 1ll * A[i + 1] % mod;
    C = multiply(C, A1);
    for (int i = n - 1; i > 0; i--) C[i] = C[i - 1] * 1ll * fpow(i, mod - 2) % mod;
    C[0] = 0;
    while (C.size() > n) C.pop_back();
    return C;
}
poly quick_pow(poly &a, int k, int k_mod_phi, bool is_k_bigger_than_mod = false) {
    assert(!a.empty());
    int n = a.size(), t = -1, b;
    for (int i = 0; i < n; i++) if (a[i]) {
        t = i, b = a[i];
        break;
    }
    if (t == -1 || t && is_k_bigger_than_mod || k * 1ll * t >= n) return poly(n, 0);
    poly f;
    for (int i = 0; i < n; i++) {
        if (i + t < n) f.push_back(a[i + t] * 1ll * fpow(b, mod - 2) % mod);
        else f.push_back(0);
    }
    f = ln(f);
    for (auto &x : f) x = x * 1ll * k % mod;
    f = exp(f);
    poly res;
    for (int i = 0; i < k * t; i++) res.push_back(0);
    int fb = fpow(b, k_mod_phi);
    for (int i = k * t; i < n; i++) res.push_back(f[i - k * t] * 1ll * fb % mod);
    return res;
}

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n, k = 0, k_mod_phi = 0, isb = 0;
    string s;
    cin >> n >> s;
    for (auto ch : s) {
        if ((ch - '0') + k * 10ll >= mod) isb = 1;
        k = ((ch - '0') + k * 10ll) % mod;
        k_mod_phi = ((ch - '0') + k_mod_phi * 10ll) % 998244352;
    }
    poly a(n);
    for (auto &x : a) cin >> x;
    a = quick_pow(a, k, k_mod_phi, isb);
    while (a.size() > n) a.pop_back();
    for (auto x : a) cout << x << ' ';
    return 0;
}
```
== 任意模数NTT

模数小于 $10^9$

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef complex<double> cp;
typedef vector<cp> poly;
typedef long long ll;

const int N = 4000000 + 5;
const double pi = acos(-1);

int rf[26][N];
void init(int n) {
    assert(n < N);
    int lg = __lg(n);
    static vector<bool> bt(26, 0);
    if (bt[lg] == 1) return;
    bt[lg] = 1;
    for (int i = 0; i < n; i++) rf[lg][i] = (rf[lg][i >> 1] >> 1) + ((i & 1) ? (n >> 1) : 0);
}
void fft(poly &x, int lim, int op) {
    int lg = __lg(lim);
    for (int i = 0; i < lim; i++) if (i < rf[lg][i]) swap(x[i], x[rf[lg][i]]);
    for (int len = 2; len <= lim; len <<= 1) {
        int k = (len >> 1);
        for (int i = 0; i < lim; i += len) {
            for (int j = 0; j < k; j++) {
                cp w(cos(pi * j / k), op * sin(pi * j / k));
                cp tmp = w * x[i + j + k];
                x[i + j + k] = x[i + j] - tmp;
                x[i + j] = x[i + j] + tmp;
            }
        }
    }
    if (op == -1) for (int i = 0; i < lim; i++) x[i] /= lim;
}
poly multiply(const poly &a, const poly &b) {
    assert(!a.empty() && !b.empty());
    int lim = 1;
    while (lim + 1 < int(a.size() + b.size())) lim <<= 1;
    init(lim);
    poly pa = a, pb = b;
    pa.resize(lim);
    pb.resize(lim);
    for (int i = 0; i < lim; i++) pa[i] = (cp){pa[i].real(), pb[i].real()};
    fft(pa, lim, 1);
    pb[0] = conj(pa[0]);
    for (int i = 1; i < lim; i++) pb[lim - i] = conj(pa[i]);
    for (int i = 0; i < lim; i++) {
        pa[i] = (pa[i] + pb[i]) * (pa[i] - pb[i]) / cp({0, 4});
    }
    fft(pa, lim, -1);
    pa.resize(int(a.size() + b.size()) - 1);
    return pa;
}
vector<int> MTT(const vector<int> &a, const vector<int> &b, const int mod) {
    const int B = (1 << 15) - 1, M = (1 << 15);
    int lim = 1;
    while (lim + 1 < int(a.size() + b.size())) lim <<= 1;
    init(lim);
    poly pa(lim), pb(lim);
    auto get = [](const vector<int>& v, int pos) -> int {
        if (pos >= v.size()) return 0;
        else return v[pos];
    };
    for (int i = 0; i < lim; i++) pa[i] = (cp){get(a, i) >> 15, get(a, i) & B};
    fft(pa, lim, 1);
    pb[0] = conj(pa[0]);
    for (int i = 1; i < lim; i++) pb[lim - i] = conj(pa[i]);
    poly A0(lim), A1(lim);
    for (int i = 0; i < lim; i++) {
        A0[i] = (pa[i] + pb[i]) / (cp){2, 0};
        A1[i] = (pa[i] - pb[i]) / (cp){0, 2}; 
    }
    for (int i = 0; i < lim; i++) pa[i] = (cp){get(b, i) >> 15, get(b, i) & B};
    fft(pa, lim, 1);
    pb[0] = conj(pa[0]);
    for (int i = 1; i < lim; i++) pb[lim - i] = conj(pa[i]);
    poly B0(lim), B1(lim);
    for (int i = 0; i < lim; i++) {
        B0[i] = (pa[i] + pb[i]) / (cp){2, 0};
        B1[i] = (pa[i] - pb[i]) / (cp){0, 2}; 
    }
    for (int i = 0; i < lim; i++) {
        pa[i] = A0[i] * B0[i];
        pb[i] = A0[i] * B1[i];
        A0[i] = pa[i];
        pa[i] = A1[i] * B1[i];
        B1[i] = pb[i];
        B0[i] = A1[i] * B0[i];
        A1[i] = pa[i];
        pa[i] = A0[i] + (cp){0, 1} * A1[i];
        pb[i] = B0[i] + (cp){0, 1} * B1[i];
    }
    fft(pa, lim, -1); fft(pb, lim, -1);
    vector<int> res(int(a.size() + b.size()) - 1);
    const int M2 = M * 1ll * M % mod;
    for (int i = 0; i < res.size(); i++) {
        ll a0 = round(pa[i].real()), a1 = round(pa[i].imag()), b0 = round(pb[i].real()), b1 = round(pb[i].imag());
        a0 %= mod; a1 %= mod; b0 %= mod; b1 %= mod;
        res[i] = (a0 * 1ll * M2 % mod + a1 + (b0 + b1) % mod * 1ll * M % mod) % mod;
    }
    return res;
}

int main() {
#ifdef LOCAL
    freopen("miku.in", "r", stdin);
    freopen("miku.out", "w", stdout);
#endif
    ios::sync_with_stdio(0); cin.tie(0);
    int n, m, p;
    cin >> n >> m >> p;
    vector<int> a(n + 1), b(m + 1);
    for (auto &x : a) cin >> x;
    for (auto &x : b) cin >> x;
    auto res = MTT(a, b, p);
    for (auto x : res) cout << x << ' ';
}
```

= 数据结构
== 李超树
```cpp
\begin{lstlisting}
struct Line {
	ll k, b;
} lin[N];
int lcnt;
int add_line(ll k, ll b) {
	lin[++lcnt] = {k, b};
	return lcnt;
}
struct node {
	int ls, rs, u;
} tr[N << 2];
int tot;
ll calc(int u, ll x) {
	return lin[u].k * x + lin[u].b;
}
bool cmp(int u, int v, ll x) {
	return calc(u, x) <= calc(v, x); // 如果要求最大值，只需要修改为大于等于
}
void pushdown(int &p, int l, int r, int v) {
	if (!p) p = ++tot;
	if (l == r) return;
	int mid = (l + r) >> 1;
	int &u = tr[p].u, b = cmp(v, u, mid);
	if (b) swap(u, v);
	int bl = cmp(v, u, l), br = cmp(v, u, r);
	if (bl) pushdown(tr[p].ls, l, mid, v);
	if (br) pushdown(tr[p].rs, mid + 1, r, v);
}
void update(int &p, int l, int r, int L, int R, int v) {
	if (l > R || r < L) return;
	if (!p) p = ++tot;
	int mid = (l + r) >> 1;
	if (l >= L && r <= R) return pushdown(p, l, r, v), void();
	update(tr[p].ls, l, mid, L, R, v);
	update(tr[p].rs, mid + 1, r, L, R, v);
}
ll query(int p, int l, int r, ll pos) {
	if (!p) return 1e16;
	ll res = calc(tr[p].u, pos);
	int mid = (l + r) >> 1;
	if (l == r) return res;
	if (pos <= mid) {
		res = min(res, query(tr[p].ls, l, mid, pos));
	} else res = min(res, query(tr[p].rs, mid + 1, r, pos));
	return res;
}

int main() {
	lin[0].b = 1e16;
	return 0;	
}
```
== 兔队线段树

求有多少个严格前缀最大值。

线段树保存每个区间为子问题时右部分的答案 res（可以不需要信息可减），和区间的最大值 mx。

calc 考虑一段区间之前有 x 大的数时，区间此时前缀最大数的树目。

1. $x >= "val"["lson"],"ans" = "calc"("rson")$

2. $x < "val"["lson"],"ans" = "calc"("lson") + "res"[p]$

```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

const int N = 1e5 + 5;
#define lson (p << 1)
#define rson ((p << 1) | 1)
#define mid ((l + r) >> 1)
int n, m;
struct node {
    int s, a, b;
} tr[N << 2];
bool cmp(int a, int b, int c, int d) {
    if (d == 0 && b == 0) return 0;
    if (d == 0 && a == 0) return 0;
    if (d == 0) return 1;
    return a * 1ll * d > c * 1ll * b; 
}
int calc(int p, int l, int r, int c, int d) {
    if (l == r) 
        return cmp(tr[p].a, tr[p].b, c, d);
    if (cmp(tr[lson].a, tr[lson].b, c, d)) {
        return calc(lson, l, mid, c, d) + tr[p].s;
    }
    return calc(rson, mid + 1, r, c, d);
}
void modify(int p, int l, int r, int pos, int v) {
    if (l == r) {
        tr[p] = {0, v, pos};
        return;
    }
    if (pos <= mid) modify(lson, l, mid, pos, v);
    else modify(rson, mid + 1, r, pos, v);
    if (cmp(tr[lson].a, tr[lson].b, tr[rson].a, tr[rson].b)) {
        tr[p] = tr[lson];
    } else tr[p] = tr[rson];
    tr[p].s = calc(rson, mid + 1, r, tr[lson].a, tr[lson].b);
}

int main() {
    scanf("%d %d", &n, &m);
    while (m--) {
        int x, y;
        scanf("%d %d", &x, &y);
        modify(1, 1, n, x, y);
        printf("%d\n", calc(1, 1, n, 0, 0));
    }
    return 0;
```

== 平衡树
```cpp
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
```

== 文艺平衡树
```cpp
# include<iostream>
# include<cstdio>
# include<cstring>
# include<cstdlib>
using namespace std;
const int MAX=1e5+1;
int n,m,tot,rt;
struct Treap{
    int pos[MAX],siz[MAX],w[MAX];
    int son[MAX][2];
    bool fl[MAX];
    void pus(int x)
    {
        siz[x]=siz[son[x][0]]+siz[son[x][1]]+1;
    }
    int build(int x)
    {
        w[++tot]=x,siz[tot]=1,pos[tot]=rand();
        return tot;
    }
    void down(int x)
    {
        swap(son[x][0],son[x][1]);
        if(son[x][0]) fl[son[x][0]]^=1;
        if(son[x][1]) fl[son[x][1]]^=1;
        fl[x]=0;
    }
    int merge(int x,int y)
    {
        if(!x||!y) return x+y;
        if(pos[x]<pos[y])
        {
            if(fl[x]) down(x);
            son[x][1]=merge(son[x][1],y);
            pus(x);
            return x;
        }
        if(fl[y]) down(y);
        son[y][0]=merge(x,son[y][0]);
        pus(y);
        return y;
    }
    void split(int i,int k,int &x,int &y)
    {
        if(!i)
        {
            x=y=0;
            return;
        }
        if(fl[i]) down(i);
        if(siz[son[i][0]]<k)
        x=i,split(son[i][1],k-siz[son[i][0]]-1,son[i][1],y);
        else
        y=i,split(son[i][0],k,x,son[i][0]);
        pus(i);
    }
    void coutt(int i)
    {
        if(!i) return;
        if(fl[i]) down(i);
        coutt(son[i][0]);
        printf("%d ",w[i]);
        coutt(son[i][1]);
    }
}Tree;
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)
      rt=Tree.merge(rt,Tree.build(i));
    for(int i=1;i<=m;i++)
      {
          int l,r,a,b,c;
          scanf("%d%d",&l,&r);
          Tree.split(rt,l-1,a,b);
        Tree.split(b,r-l+1,b,c);
        Tree.fl[b]^=1;
        rt=Tree.merge(a,Tree.merge(b,c));
      }
    Tree.coutt(rt);
    return 0;
}
```

= 字符串
== KMP
```cpp
int n = strlen(s + 1);
for (int i = 2; i <= n; i++) {
    int j = k[i - 1];
    while (j != 0 && s[i] != s[j + 1]) j = k[j];
    if (s[i] == s[j + 1]) k[i] = j + 1;
    else k[i] = 0;
}
```
== Z function
```cpp
 for (int i = 2, l = 0, r = 0; i <= n; i++) {
     if (r >= i && r - i + 1 > z[i - l + 1]) {
     	z[i] = z[i - l + 1];
     } else {
     	z[i] = max(0, r - i + 1);
     	while (z[i] < n - i + 1 && s[z[i] + 1] == s[i + z[i]]) ++z[i];
     }
     if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
 }
```
== SA
```cpp
int sa[N], ork[N], rk[N], cnt[N], id[N], h[N], M, n;
char s[N];
int mn[22][N];
int lcp(int a, int b) {
    if (a == b) return n - a + 1;
    if (rk[a] > rk[b]) swap(a, b);
    int l = rk[a] + 1, r = rk[b];
    int len = r - l + 1, k = __lg(len);
    return min(mn[k][l], mn[k][r - (1 << k) + 1]);
}
void MAIN() {
    scanf("%s", s + 1);
    n = strlen(s + 1);
    for (int i = 1; i <= n; i++) M = max(M, (int)s[i]);
    for (int i = 1; i <= n; i++) if ((int)(s[i]) > M) M = (int)(s[i]);
    for (int i = 1; i <= n; i++) cnt[rk[i] = s[i]]++;
    for (int i = 0; i <= M; i++) cnt[i] += cnt[i - 1];
    for (int i = n; i; i--) sa[cnt[rk[i]]--] = i;
    for (int w = 1, p; w < n; w <<= 1, M = p) {
        p = 0;
        for (int i = n; i > n - w; i--) id[++p] = i;
        for (int i = 1; i <= n; i++) if (sa[i] > w) id[++p] = sa[i] - w;
        for (int i = 0; i <= M; i++) cnt[i] = 0;
        for (int i = 1; i <= n; i++) cnt[rk[i]]++;
        for (int i = 1; i <= M; i++) cnt[i] += cnt[i - 1];
        for (int i = n; i; i--) sa[cnt[rk[id[i]]]--] = id[i];
        p = 0;
        for (int i = 0; i <= n; i++) ork[i] = rk[i];
        for (int i = 1; i <= n; i++) {
            if (ork[sa[i]] == ork[sa[i - 1]] && ork[sa[i] + w] == ork[sa[i - 1] + w]) rk[sa[i]] = p;
            else rk[sa[i]] = ++p;
        }
        if (p == n) break;
    }
    for (int i = 1, k = 0; i <= n; i++) {
        if (rk[i] == 1) continue;
        if (k) k--;
        while (s[i + k] == s[sa[rk[i] - 1] + k]) k++;
        h[rk[i]] = k;
    }
    for (int i = 1; i <= n; i++) mn[0][i] = h[i];
    for (int j = 1; j < 22; j++) {
        for (int i = 1; i <= n; i++) {
            mn[j][i] = min(mn[j - 1][i], mn[j - 1][min(n, i + (1 << (j - 1)))]);
        }
    }
}
```
== AC自动机
```cpp
int ch[N][26], tot, fail[N], e[N];
void insert(const char *s) {
	int u = 0, n = strlen(s + 1);
	for (int i = 1; i <= n; i++) {
        if (!ch[u][s[i] - 'a']) ch[u][s[i] - 'a'] = ++tot;
        u = ch[u][s[i] - 'a'];
	}
	e[u] += 1;
}
void build() {
	queue<int> q;
	for (int i = 0; i <= 25; i++) if (ch[0][i]) q.push(ch[0][i]);
	while (!q.empty()) {
		int now = q.front(); q.pop();
		for (int i = 0; i < 26; i++) {
			if (ch[now][i]) fail[ch[now][i]] = ch[fail[now]][i], q.push(ch[now][i]);
			else ch[now][i] = ch[fail[now]][i];
		}
	}
}
int query(const char *s) {
    int u = 0, n = strlen(s + 1), res = 0;
    for (int i = 1; i <= n; i++){
    	u = ch[u][s[i] - 'a'];
    	for (int j = u; j && e[j] != -1; j = fail[j]) {
    		res += e[j];
    		e[j] = -1;
    	}
    }
    return res;
}
```
== Manacher

对于第 $i$ 个字符为对称轴: 

+ 如果回文串长为奇数, $d[2 * i]/2$ 是半径加上自己的长度

+ 如果长为偶数, $d[2 * i -1]/2$ 是半径的长度, 方向向右. 

```cpp
int n, d[N * 2];
char s[N];

for (int i = 1; i <= n; i++) t[i * 2] = s[i], t[i * 2 - 1] = '#';
t[n * 2 + 1] = '#';
m = n * 2 + 1;
for (int i = 1, l = 0, r = 0; i <= m; i++) {
  	int k = i <= r ? min(d[r - i + l], r - i + 1) : 1;
   	while (i + k <= m && i - k >= 1 && t[i + k] == t[i - k]) k++;
   	d[i] = k--;
   	if (i + k > r) r = i + k, l = i - k;
}
```

= 杂项
== fastio

来自 oiwiki

```cpp
// #define DEBUG 1  // 调试开关
struct IO {
#define MAXSIZE (1 << 20)
#define isdigit(x) (x >= '0' && x <= '9')
  char buf[MAXSIZE], *p1, *p2;
  char pbuf[MAXSIZE], *pp;
#if DEBUG
#else
  IO() : p1(buf), p2(buf), pp(pbuf) {}

  ~IO() { fwrite(pbuf, 1, pp - pbuf, stdout); }
#endif
  char gc() {
#if DEBUG  // 调试，可显示字符
    return getchar();
#endif
    if (p1 == p2) p2 = (p1 = buf) + fread(buf, 1, MAXSIZE, stdin);
    return p1 == p2 ? ' ' : *p1++;
  }

  bool blank(char ch) {
    return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t';
  }

  template <class T>
  void read(T &x) {
    double tmp = 1;
    bool sign = false;
    x = 0;
    char ch = gc();
    for (; !isdigit(ch); ch = gc())
      if (ch == '-') sign = 1;
    for (; isdigit(ch); ch = gc()) x = x * 10 + (ch - '0');
    if (ch == '.')
      for (ch = gc(); isdigit(ch); ch = gc())
        tmp /= 10.0, x += tmp * (ch - '0');
    if (sign) x = -x;
  }

  void read(char *s) {
    char ch = gc();
    for (; blank(ch); ch = gc());
    for (; !blank(ch); ch = gc()) *s++ = ch;
    *s = 0;
  }

  void read(char &c) { for (c = gc(); blank(c); c = gc()); }

  void push(const char &c) {
#if DEBUG  // 调试，可显示字符
    putchar(c);
#else
    if (pp - pbuf == MAXSIZE) fwrite(pbuf, 1, MAXSIZE, stdout), pp = pbuf;
    *pp++ = c;
#endif
  }

  template <class T>
  void write(T x) {
    if (x < 0) x = -x, push('-');  // 负数输出
    static T sta[35];
    T top = 0;
    do {
      sta[top++] = x % 10, x /= 10;
    } while (x);
    while (top) push(sta[--top] + '0');
  }

  template <class T>
  void write(T x, char lastChar) {
    write(x), push(lastChar);
  }
} io;

```
== 高精度

来自 oiwiki

```cpp
constexpr int MAXN = 9999;
// MAXN 是一位中最大的数字
constexpr int MAXSIZE = 10024;
// MAXSIZE 是位数
constexpr int DLEN = 4;

// DLEN 记录压几位
struct Big {
  int a[MAXSIZE], len;
  bool flag;  // 标记符号'-'

  Big() {
    len = 1;
    memset(a, 0, sizeof a);
    flag = false;
  }

  Big(const int);
  Big(const char*);
  Big(const Big&);
  Big& operator=(const Big&);
  Big operator+(const Big&) const;
  Big operator-(const Big&) const;
  Big operator*(const Big&) const;
  Big operator/(const int&) const;
  // TODO: Big / Big;
  Big operator^(const int&) const;
  // TODO: Big ^ Big;

  // TODO: Big 位运算;

  int operator%(const int&) const;
  // TODO: Big ^ Big;
  bool operator<(const Big&) const;
  bool operator<(const int& t) const;
  void print() const;
};

Big::Big(const int b) {
  int c, d = b;
  len = 0;
  // memset(a,0,sizeof a);
  CLR(a);
  while (d > MAXN) {
    c = d - (d / (MAXN + 1) * (MAXN + 1));
    d = d / (MAXN + 1);
    a[len++] = c;
  }
  a[len++] = d;
}

Big::Big(const char* s) {
  int t, k, index, l;
  CLR(a);
  l = strlen(s);
  len = l / DLEN;
  if (l % DLEN) ++len;
  index = 0;
  for (int i = l - 1; i >= 0; i -= DLEN) {
    t = 0;
    k = i - DLEN + 1;
    if (k < 0) k = 0;
    g(j, k, i) t = t * 10 + s[j] - '0';
    a[index++] = t;
  }
}

Big::Big(const Big& T) : len(T.len) {
  CLR(a);
  f(i, 0, len) a[i] = T.a[i];
  // TODO:重载此处？
}

Big& Big::operator=(const Big& T) {
  CLR(a);
  len = T.len;
  f(i, 0, len) a[i] = T.a[i];
  return *this;
}

Big Big::operator+(const Big& T) const {
  Big t(*this);
  int big = len;
  if (T.len > len) big = T.len;
  f(i, 0, big) {
    t.a[i] += T.a[i];
    if (t.a[i] > MAXN) {
      ++t.a[i + 1];
      t.a[i] -= MAXN + 1;
    }
  }
  if (t.a[big])
    t.len = big + 1;
  else
    t.len = big;
  return t;
}

Big Big::operator-(const Big& T) const {
  int big;
  bool ctf;
  Big t1, t2;
  if (*this < T) {
    t1 = T;
    t2 = *this;
    ctf = true;
  } else {
    t1 = *this;
    t2 = T;
    ctf = false;
  }
  big = t1.len;
  int j = 0;
  f(i, 0, big) {
    if (t1.a[i] < t2.a[i]) {
      j = i + 1;
      while (t1.a[j] == 0) ++j;
      --t1.a[j--];
      // WTF?
      while (j > i) t1.a[j--] += MAXN;
      t1.a[i] += MAXN + 1 - t2.a[i];
    } else
      t1.a[i] -= t2.a[i];
  }
  t1.len = big;
  while (t1.len > 1 && t1.a[t1.len - 1] == 0) {
    --t1.len;
    --big;
  }
  if (ctf) t1.a[big - 1] = -t1.a[big - 1];
  return t1;
}

Big Big::operator*(const Big& T) const {
  Big res;
  int up;
  int te, tee;
  f(i, 0, len) {
    up = 0;
    f(j, 0, T.len) {
      te = a[i] * T.a[j] + res.a[i + j] + up;
      if (te > MAXN) {
        tee = te - te / (MAXN + 1) * (MAXN + 1);
        up = te / (MAXN + 1);
        res.a[i + j] = tee;
      } else {
        up = 0;
        res.a[i + j] = te;
      }
    }
    if (up) res.a[i + T.len] = up;
  }
  res.len = len + T.len;
  while (res.len > 1 && res.a[res.len - 1] == 0) --res.len;
  return res;
}

Big Big::operator/(const int& b) const {
  Big res;
  int down = 0;
  gd(i, len - 1, 0) {
    res.a[i] = (a[i] + down * (MAXN + 1)) / b;
    down = a[i] + down * (MAXN + 1) - res.a[i] * b;
  }
  res.len = len;
  while (res.len > 1 && res.a[res.len - 1] == 0) --res.len;
  return res;
}

int Big::operator%(const int& b) const {
  int d = 0;
  gd(i, len - 1, 0) d = (d * (MAXN + 1) % b + a[i]) % b;
  return d;
}

Big Big::operator^(const int& n) const {
  Big t(n), res(1);
  int y = n;
  while (y) {
    if (y & 1) res = res * t;
    t = t * t;
    y >>= 1;
  }
  return res;
}

bool Big::operator<(const Big& T) const {
  int ln;
  if (len < T.len) return true;
  if (len == T.len) {
    ln = len - 1;
    while (ln >= 0 && a[ln] == T.a[ln]) --ln;
    if (ln >= 0 && a[ln] < T.a[ln]) return true;
    return false;
  }
  return false;
}

bool Big::operator<(const int& t) const {
  Big tee(t);
  return *this < tee;
}

void Big::print() const {
  printf("%d", a[len - 1]);
  gd(i, len - 2, 0) { printf("%04d", a[i]); }
}

void print(const Big& s) {
  int len = s.len;
  printf("%d", s.a[len - 1]);
  gd(i, len - 2, 0) { printf("%04d", s.a[i]); }
}
```

== 手写 bitset
```cpp
struct Bitset {
    #define For(i,a,b) for(int i=a,i##end=b; i<=i##end; i++)
    #define foR(i,a,b) for(int i=a,i##end=b; i>=i##end; i--)
    using uint = unsigned int;
    using ull = unsigned long long;
    vector < ull > bit; int len;
    Bitset(int x = n) {x = (x >> 6) + 1; bit.resize(x); len = x;}
    void resize(int x) {bit.resize((x >> 6) + 1); len = (x >> 6) + 1;For(i, 0, len-1) bit[i] = 0;}
    void set1(int x) {bit[x>>6] |= (1ull<<(x&63));}
    void set0(int x) {bit[x>>6] &= (~(1ull<<(x&63)));}
    void flip(int x) {bit[x>>6] ^= (1ull<<(x&63));}
    bool operator [] (int x) {return (bit[x>>6] >> (x&63)) & 1;}
    bool any() {For(i, 0, len-1) if(bit[i]) return 1;return 0;}
    Bitset operator ~ () const {Bitset res(len);For(i, 0, len-1) res.bit[i] = ~bit[i];return res;}
    Bitset operator | (const Bitset &b) const {Bitset res(len); For(i, 0, len-1) res.bit[i] = bit[i] | b.bit[i];return res;}
    Bitset operator & (const Bitset &b) const {Bitset res(len); For(i, 0, len-1) res.bit[i] = bit[i] & b.bit[i];return res;}
    Bitset operator ^ (const Bitset &b) const {Bitset res(len); For(i, 0, len-1) res.bit[i] = bit[i] ^ b.bit[i];return res;}
    void operator &= (const Bitset &b) {For(i, 0, len-1) bit[i] &= b.bit[i];}
    void operator |= (const Bitset &b) {For(i, 0, len-1) bit[i] |= b.bit[i];}
    void operator ^= (const Bitset &b) {For(i, 0, len-1) bit[i] ^= b.bit[i];}
    Bitset operator << (const int t) const {
        Bitset res(len); int high = t >> 6, low = t & 63; ull lst = 0;
        for(int i = 0; i + high < len; i++) {
            res.bit[i + high] = (lst | (bit[i] << low));
            if(low) lst = (bit[i] >> (64 - low));
        }
        return res;
    }
    Bitset operator >> (const int t) const {
        Bitset res(len); int high = t >> 6, low = t & 63; ull lst = 0;
        for(int i = len - 1; i >= high; i--) {
            res.bit[i - high] = (lst | (bit[i] >> low));
            if(low) lst = (bit[i] << (64 - low));
        }
        return res;
    }
    void operator <<= (const int t) {
        int high = t >> 6, low = t & 63;
        for(int i = len - high - 1; ~i; i--) {
            bit[i + high] = (bit[i] << low);
            if(low && i) bit[i + high] |= (bit[i - 1] >> (64 - low));
        }
        for(int i = 0; i < min(high, len - 1); i++) bit[i] = 0;
    } 
    void operator >>= (const int t) {
        int high = t >> 6, low = t & 63;
        for(int i = high; i < len; i++) {
            bit[i - high] = (bit[i] >> low);
            if(low && i != len) bit[i - high] |= (bit[i + 1] << (64 - low));
        }
        for(int i = max(len - high, 0); i < len; i++) bit[i] = 0;
    } 
    ull get(int x) {
        int t = x >> 6, q = x & 63;
        if (q == 63) return bit[t];
        return bit[t] & ((1ull << (q + 1)) - 1);
    }
    ull get(int l, int r) {
        int lt = (l >> 6), rt = (r >> 6);
        if (lt == rt) {
            if ((l & 63) == 0) return get(r);
            return (get(r) - get(l - 1)) >> ((l & 63));
        }
         ull a = (l & 63) == 0 ? (bit[lt]) : ((bit[lt] - get(l - 1)) >> ((l & 63)));
         return a + (get(r) << (64 - (l & 63)));
     }
}
```

== 对拍
```bash
#!/usr/bin/bash
g++ ./my.cpp -o my -std=c++17 -fsanitize=undefined
g++ ./std.cpp -o std -std=c++17 -fsanitize=undefined
g++ ./data.cpp -o data -std=c++17 -fsanitize=undefined
cnt=0;
while true; do
	./data > data.in
	./my < data.in > my.out
	./std < data.in > std.out
	if diff my.out std.out; then
		let cnt++;
		echo "# $cnt AC";
	else
		echo "WA";
		break;
	fi
done
```