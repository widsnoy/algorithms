bool gauss() {
    int k = 1;
    for (int i = 1; i <= m; i++) {
        if (k > n) break;
        if (a[k][i] == 0) {
            for (int j = k + 1; j <= n; j++) if (a[j][i] != 0) {
                for (int l = 1; l <= m + 1; l++) swap(a[j][l], a[k][l]);
                break;
            }
        }
        if (a[k][i] == 0) continue;
        for (int j = k + 1; j <= n; j++) if (a[j][i] == 1) {
            for (int l = i; l <= m + 1; l++) a[j][l] ^= a[k][l];
        }
        k++;
    }
    int flag = 1;
    for (int i = k; i <= n; i++) if (a[i][m + 1] == 1) flag = 0;
    return flag;
}