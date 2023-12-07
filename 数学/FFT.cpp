typedef long long ll;
typedef complex<double> cp; 
const int N = 6e5 + 5;
const double pi = acos(-1.0);
int n, m, len = 1, l, rev[N], x[N], y[N];
cp a[N * 2], b[N];

void fft(cp *a, int n, int inv) {
    for (int i = 0; i < n; i++) if (rev[i] < i) swap(a[i], a[rev[i]]);
    for (int k = 1; k < n; k <<= 1) {
        cp wn(cos(pi / k), inv * sin(pi / k));
        for (int i = 0; i < n; i += k * 2) {
            cp w(1, 0);
            for (int j = 0; j < k; j++, w *= wn) {
                cp x = a[i + j], y = a[i + j + k] * w;
                a[i + j] = x + y, a[i + j + k] = x - y;
            }
        }
    }
    if (inv < 0) for (int i = 0; i < len; i++) a[i] /= n;
}
