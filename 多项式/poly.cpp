#include <bits/stdc++.h>
using namespace std;

typedef vector<int> poly;
const int mod = 998244353;
const int N = 4000000 + 5;

int rf[N];
int fpow(int a, int b) {
    int res = 1;
    for (; b; b >>= 1, a = a * 1ll * a % mod) if (b & 1)
        res = res * 1ll * a % mod;
    return res;
}
void init(int n) {
    static int mem = -1;
    if (n == mem) return;
    mem = n;
    assert(n < N);
    for (int i = 0; i < n; i++) rf[i] = (rf[i >> 1] >> 1) + ((i & 1) ? (n >> 1) : 0);
}
void ntt(poly &x, int lim, int op) {
    for (int i = 0; i < lim; i++) if (i < rf[i]) swap(x[i], x[rf[i]]);
    int gn, g, tmp;
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
poly prod_poly(const vector<poly>& vec) {
    int n = vec.size();
    auto calc = [&](const auto &self, int l, int r) -> poly {
        if (l == r) return vec[l];
        int mid = (l + r) >> 1;
        return multiply(calc(l, mid), calc(mid + 1, r));
    }
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
    return res;
}
poly exp(const poly &g) {
    int n = g.size();
    poly b(n, 0);
    for (int i = 1; i < n; i++) b[i] = i * 1ll * g[i] % mod;
    return semi_online_convolution(b, n, 1);
}

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    return 0;
}