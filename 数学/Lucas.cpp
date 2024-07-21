#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 2e5;
int fac[N], ifac[N], mod;
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b != 0) {
        ll g = exgcd(b, a % b, y, x);
        return y -= a / b * x, g;
    } return x = 1, y = 0, a;
}
int fpow(int a, int b) {
    int res = 1;
    for (; b; b >>= 1, a = a * 1ll * a % mod) if (b & 1) res = res * 1ll * a % mod;
    return res;
}
ll getinv(ll a, ll mod) {
    return fpow(a, mod - 2);
    ll x, y;
    exgcd(a, mod, x, y);
    x = (x % mod + mod) % mod;
    return x;
}
void init_binom(int n) {
    fac[0] = ifac[0] = 1;
    for (int i = 1; i <= n; i++) fac[i] = fac[i - 1] * 1ll * i % mod;
    ifac[n] = getinv(fac[n], mod);
    for (int i = n; i > 1; i--) ifac[i - 1] = ifac[i] * 1ll * i % mod;
}
int binom(int a, int b) {
    if (b < 0 || a < 0 || b > a) return 0;
    return fac[a] * 1ll * ifac[b] % mod * ifac[a - b] % mod; 
}
int lucas(int a, int b) {
    if (a < mod) return binom(a, b);
    return lucas(a / mod, b / mod) * 1ll * binom(a % mod, b % mod) % mod;
}
int main() {
    int T;
    cin >> T;
    while (T--) {
        int n, m;
        cin >> n >> m >> mod;
        init_binom(mod - 1);
        cout << lucas(n + m, m) << '\n';
    }
    return 0;
}