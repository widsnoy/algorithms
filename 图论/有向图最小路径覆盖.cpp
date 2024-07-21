int n, m;
bitset<N> f[N];
int vis[N], mch[N];

bool dfs(int u, int dfc) {
    for (int v = 1; v <= n; v++) if (v != u && vis[v] != dfc && f[u][v]) {
        vis[v] = dfc;
        if (!mch[v] || dfs(mch[v], dfc)) return mch[v] = u, 1;
    }
    return 0;
}

void solve() {
    memset(vis, 0, sizeof vis);
    memset(mch, 0, sizeof mch);
    for (int i = 1; i <= n; i++) f[i].reset();
    for (int i = 1; i <= m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        f[u].set(v);
    }
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) if (f[i][k]) f[i] |= f[k];
    }
    int res = n;
    for (int i = 1; i <= n; i++) res -= dfs(i, i);
    printf("%d\n", res); 
}
