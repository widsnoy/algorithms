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
