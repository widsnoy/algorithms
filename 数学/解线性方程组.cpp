void gauss() {
    for (int i = 0; i < n; i++) {
        int id = i;
        for (int j = i + 1; j < n; j++) if (fabs(a[j][i]) > fabs(a[id][i])) id = j;
        for (int j = i; j <= n; j++) swap(a[id][j], a[i][j]);
        if (a[i][i] == 0) {
            puts("No Solution");
            return;
        }
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double t = a[j][i] / a[i][i];
            for (int k = i; k <= n; k++) a[j][k] -= a[i][k] * t;
        }
    }
    for (int i = 0; i < n; i++) printf("%.2lf\n", a[i][n] / a[i][i]);
} 
