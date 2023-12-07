// 倍增
int faz[N][20], dep[N];
void dfs(int u, int fa) {
    faz[u][0] = fa;
    dep[u] = dep[fa] + 1;
    for (int i = 1; i < 20; i++) faz[u][i] = faz[faz[u][i - 1]][i - 1];
    for (int v : G[u]) if (v != fa) {
        dfs(v, u);
    }
}
int LCA(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);
    int d = dep[u] - dep[v];
    for (int i = 0; i < 20; i++) if ((d >> i) & 1) u = faz[u][i];
    if (v == u) return u;
    for (int i = 19; i >= 0; i--) if (faz[u][i] != faz[v][i]) 
        u = faz[u][i], v = faz[v][i];
    return faz[u][0];
}

//树剖
int dfc, dfn[N], rnk[N], siz[N], top[N], dep[N], son[N], faz[N];
void dfs1(int u, int fa) {
    dep[u] = dep[fa] + 1;
    siz[u] = 1;
    son[u] = -1;
    faz[u] = fa;
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs1(v, u);
        siz[u] += siz[v];
        if (son[u] == -1 || siz[son[u]] < siz[v]) son[u] = v;
    }
}
void dfs2(int u, int fa, int tp) {
    dfn[u] = ++dfc;
    rnk[dfc] = u;
    top[u] = tp;
    if (son[u] != -1) dfs2(son[u], u, tp);
    for (int v : G[u]) {
        if (v == fa || v == son[u]) continue;
        dfs2(v, u, v);
    }
}
int LCA(int u, int v) {
    while (top[u] != top[v]) {
        if (dep[top[u]] > dep[top[v]])
            u = faz[top[u]];
        else
            v = faz[top[v]];
    }
    return dep[u] > dep[v] ? v : u;
}