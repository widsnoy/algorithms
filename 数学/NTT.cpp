#include <bits/stdc++.h>
using namespace std;

const int N = 4e6 + 5, mod = 998244353;
int n, m, r[N], lim, a[N], b[N];

int fpow(int a, int b) {
	int res = 1;
	for (; b; b >>= 1, a = a * 1ll * a % mod) if (b & 1) res = res * 1ll * a % mod;
	return res;
}
void ntt(int *x, int lim, int op) {
	int i, j, k, m, gn, g, tmp;
	for (int i = 0; i < lim; i++) if (r[i] < i) swap(x[i], x[r[i]]);
	for (m = 2; m <= lim; m <<= 1) {
		k = m >> 1;
		gn = fpow(3, (mod - 1) / m);
		for (i = 0; i < lim; i += m) {
			g = 1;
			for (j = 0; j < k; j++, g = g * 1ll * gn % mod) {
				tmp = x[i + j + k] * 1ll * g % mod;
				x[i + j + k] = (x[i + j] - tmp + mod) % mod;
				x[i + j] = (x[i + j] + tmp) % mod;
			}
		}
	}
	if (op == -1) {
		reverse(x + 1, x + lim);
		int inv = fpow(lim, mod - 2);
		for (int i = 0; i < lim; i++) x[i] = x[i] * 1ll * inv % mod;
	}
}

int main() {
    scanf("%d %d", &n, &m);
    for (int i = 0; i <= n; i++) scanf("%d", &a[i]);
    for (int i = 0; i <= m; i++) scanf("%d", &b[i]);
    lim = 1;
    while (lim < (n + m) << 1) lim <<= 1;
    for (int i = 0; i < lim; i++) r[i] = (i & 1) * (lim >> 1) + (r[i >> 1] >> 1);
    ntt(a, lim, 1), ntt(b, lim, 1);
    for (int i = 0; i < lim; i++) a[i] = a[i] * 1ll * b[i] % mod;
    ntt(a, lim, -1);
    for (int i = 0; i <= n + m; i++) printf("%d ", a[i]);
    return puts(""), 0;
}
