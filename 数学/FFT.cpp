#include <bits/stdc++.h>
using namespace std;

typedef complex<double> cp;
const int N = 4e6 + 5;
const double pi = acos(-1.0);
int n, m, limit = 1, l, rev[N];
cp a[N], b[N];

void FFT(cp *a, int n, int inv) {
    for (int i = 0; i < n; i++) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int k = 1; k < n; k <<= 1) {
    	cp wn(cos(pi / k), inv * sin(pi / k));
    	for (int i = 0; i < n; i = i + k + k) {
            cp w(1, 0);
    		for (int j = 0; j < k; j++, w *= wn) {
    			cp x = a[i + j], y = w * a[i + j + k];
    			a[i + j] = x + y;
    			a[i + j + k] = x - y;
    		}
    	}
    }
    if (inv < 0) for (int i = 0; i < n; i++) a[i] /= n;
}

int main() {
    scanf("%d %d", &n, &m);
    for (int i = 0; i <= n; i++) scanf("%lf", &a[i]);
    for (int j = 0; j <= m; j++) scanf("%lf", &b[j]);
    while (n + m >= limit) limit <<= 1, l = l + 1;
    for (int i = 1; i <= limit; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (l - 1));
    FFT(a, limit, 1), FFT(b, limit, 1);
    for (int i = 0; i < limit; i++) a[i] *= b[i];
    FFT(a, limit, -1);
    for (int i = 0; i <= n + m; i++) printf("%d ", (int)(a[i].real() + 0.5));
    return 0;
}