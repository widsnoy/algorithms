int fpow(int a, int b) {
    int res = 1;
    for (; b; b >>= 1, a = a * 1ll * a % mod) if (b & 1) res = res * 1ll * a % mod;
    return res;
}
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b) {
        ll d = exgcd(b, a % b, y, x);
        return y -= a / b * x, d;
    } else return x = 1, y = 0, a;
}
int getinv(int v) {
    return fpow(v, mod - 2);
    // ll x, y;
    // exgcd(v, mod, x, y);
    // return (x % mod + mod) % mod;
}
int fac[N], ifac[N];
void init_binom(int n) {
    fac[0] = ifac[0] = 1;
    for (int i = 1; i <= n; i++) fac[i] = fac[i - 1] * 1ll * i % mod;
    ifac[n] = getinv(fac[n]);
    for (int i = n; i > 1; i--) ifac[i - 1] = ifac[i] * 1ll * i % mod;
}
int binom(int a, int b) {
    if (b < 0 || a < 0 || b > a) return 0;
    return fac[a] * 1ll * ifac[b] % mod * ifac[a - b] % mod; 
}
int getphi(int x) {
    int res = 1;
    for (int i = 2; i * i <= x; i++) {
        if (x % i == 0) {
            x /= i;
            res *= (i - 1);
            while (x % i == 0) {
                x /= i;
                res *= i;
            }
        }
    }
    if (x > 1) res *= (x - 1);
    return res;
}
int prime[N], pcnt;
bool isp[N];
int get_prime(int n) {
    for (int i = 2; i <= n; i++) {
        if (!isp[i]) prime[++pcnt] = i;
        for (int j = 1; j <= pcnt && i * prime[j] <= n; j++) {
            isp[prime[j] * i] = 1;
            if (i % prime[j] == 0) break;
        }
    }
} 
pll get_up(ll a, ll b, ll x1, ll x2) {
    //x2>=ax+b>=x1 a >= 0
    if (a == 0) return (b >= x1 && b <= x2) ? (pll){-1e18, 1e18} : (pll){1, 0};
    ll L, R;
    ll l = (x1 - b) / a - 3;
    for (L = l; L * a + b < x1; L++);
    ll r = (x2 - b) / a + 3;
    for (R = r; R * a + b > x2; R--);
    return {L, R}; 
}
pll get_dn(ll a, ll b, ll x1, ll x2) {
    //x2>=ax+b>=x1 a <= 0
    if (a == 0) return (b >= x1 && b <= x2) ? (pll){-1e18, 1e18} : (pll){1, 0};
    ll L, R;
    ll l = (x2 - b) / a - 3;
    for (L = l; L * a + b > x2; L++);
    ll r = (x1 - b) / a + 3;
    for (R = r; R * a + b < x1; R--);
    return {L, R}; 
}