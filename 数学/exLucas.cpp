#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e6;
ll a1, b1, mod;
ll m[N], a[N], pr[N], tot;
ll pre[N];
ll fpow(ll a, ll b, ll p) {
    ll res = 1;
    for (; b; b >>= 1, a = a * a % p) if (b & 1) res = res * a % p;
    return res;
}
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
ll F(ll n, int k) {
    if (n == 0) return 1;
    ll res = fpow(pre[m[k]], n / m[k], m[k]), rem = n % m[k];
    res = res * pre[rem] % m[k];
    return F(n / pr[k], k) * res % m[k];
}
int G(ll n, ll p) {
    if (n < p) return 0;
    return G(n / p, p) + n / p;
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
int main() {
    cin >> a1 >> b1 >> mod;
    ll x = mod;
    for (ll i = 2; i * i <= x; i++) {
        if (x % i) continue;
        pr[++tot] = i;
        m[tot] = 1;
        while (x % i == 0) x /= i, m[tot] *= i;
    }
    if (x != 1) pr[++tot] = x, m[tot] = x; 
    for (int k = 1; k <= tot; k++) {
        pre[0] = 1;
        for (int i = 1; i <= m[k]; i++) pre[i] = pre[i - 1] * (i % pr[k] == 0 ? 1 : i) % m[k];
        ll res = F(a1, k) * getinv(F(b1, k), m[k]) % m[k] * getinv(F(a1 - b1, k), m[k]) % m[k];
        ll d = G(a1, pr[k]) - G(b1, pr[k]) - G(a1 - b1, pr[k]), r = (d < 0 ? getinv(fpow(pr[k], -d, m[k]), m[k]) : fpow(pr[k], d, m[k]));
        res = res * r % mod;
        a[k] = res;
    }
    ll ans = 0;
    for (int i = 1; i <= tot; i++) {
        ll p = mod / m[i], q = getinv(p, m[i]);
        ans += mul(p, mul(q, a[i], mod), mod);
        ans %= mod;
    }
    cout << ans << '\n';
    return 0;
}