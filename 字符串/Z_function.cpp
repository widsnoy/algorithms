 for (int i = 2, l = 0, r = 0; i <= n; i++) {
     if (r >= i && r - i + 1 > z[i - l + 1]) {
     	z[i] = z[i - l + 1];
     } else {
     	z[i] = max(0, r - i + 1);
     	while (z[i] < n - i + 1 && s[z[i] + 1] == s[i + z[i]]) ++z[i];
     }
     if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
 }
