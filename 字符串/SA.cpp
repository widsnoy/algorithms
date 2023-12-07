const int N = 2e5 + 5;
int sa[N << 1], ork[N << 1], rk[N << 1], cnt[N], id[N << 1], M, n;
char s[N];

int main() {
    scanf("%s", s + 1);
    n = strlen(s + 1);
    for (int i = n + 1; i <= (n << 1); i++) s[i] = s[i - n], M = max(M, (int)s[i]);
    n <<= 1;
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
    return 0;
}
