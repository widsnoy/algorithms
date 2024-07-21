#define fi first
#define se second
typedef long long ll;
typedef pair<ll, ll> pll;
typedef long double ld;
//std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
#define y1 miku
const int mod = 998244353;
const int N = 1e5 + 5;
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b) {
        ll d = exgcd(b, a % b, y, x);
        return y -= a / b * x, d;
    } return x = 1, y = 0, a;
}
pll get_up(ll a, ll b, ll x1, ll x2) {
    //x2>=ax+b>=x1
    if (a == 0) return (b >= x1 && b <= x2) ? (pll){0, min(n, m)} : (pll){1, 0};
    ll L, R;
    ll l = (x1 - b) / a - 3;
    for (L = l; L * a + b < x1; L++);
    ll r = (x2 - b) / a + 3;
    for (R = r; R * a + b > x2; R--);
    return {L, R}; 
}
pll get_dn(ll a, ll b, ll x1, ll x2) {
    //x2>=ax+b>=x1
    if (a == 0) return (b >= x1 && b <= x2) ? (pll){0, min(n, m)} : (pll){1, 0};
    ll L, R;
    ll l = (x2 - b) / a - 3;
    for (L = l; L * a + b > x2; L++);
    ll r = (x1 - b) / a + 3;
    for (R = r; R * a + b < x1; R--);
    return {L, R}; 
}
//ax+b+c=0 [x1,x2] [y1,y2]
ll solve(ll a, ll b, ll c, ll x1, ll x2, ll y1, ll y2) {
    if (a == 0 && b == 0) return (c == 0) * (y2 - y1 + 1) * (x2 - x1 + 1);
    ll x, y, d = exgcd(a, b, x, y);
    if (c % d != 0) return 0;
    x *= c / d, y *= c / d;
    ll sx = b / d, sy = -a / d;
    auto A = (sx > 0 ? get_up(sx, x, x1, x2) : get_dn(sx, x, x1, x2));
    auto B = (sy > 0 ? get_up(sy, y, y1, y2) : get_dn(sy, y, y1, y2));
    A.fi = max(A.fi, B.fi), A.se = min(A.se, B.se);
    return max(0ll, A.se - A.fi + 1);
}