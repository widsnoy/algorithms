const int N = 5000 + 5;
int n, m, stk[N], top, ccno, sc[N];
int dfn[N], dfc, low[N];
int mp[N][N];
int in[N];
int head[N], ecnt;
struct Edge {
    int nxt, v;
} e[N << 2];
void add_edge(int u, int v) {
    e[ecnt] = {head[u], v}; head[u] = ecnt++;
    e[ecnt] = {head[v], u}; head[v] = ecnt++;
}
void dfs(int u, int from) {
    stk[++top] = u;
    low[u] = dfn[u] = ++dfc;
    for (int i = head[u]; i != -1; i = e[i].nxt) {
        int v = e[i].v;
        if (!dfn[v]) {
            dfs(v, i);
            low[u] = min(low[u], low[v]);
        } else if ((i ^ 1) != from) low[u] = min(low[u], dfn[v]);
    }
    if (dfn[u] == low[u]) {
        ccno++;
        int x;
        while (true) {
            x = stk[top--];
            sc[x] = ccno;
            if (x == u) break;
        }
    }
}

void solve() {
    memset(head, -1, sizeof head);
    scanf("%d %d", &n, &m);
    for (int i = 1; i <= m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        add_edge(u, v);
    }
    for (int i = 1; i <= n; i++) if (!dfn[i]) dfs(i, i);
    for (int i = 1; i <= n; i++) {
        for (int k = head[i]; k != -1; k = e[k].nxt) {
            int j = e[k].v;
            if (sc[i] != sc[j]) mp[sc[i]][sc[j]] = 1;
        }
    }
    
    for (int i = 1; i <= ccno; i++) {
        for (int j = 1; j <= ccno; j++) if (mp[i][j]) in[j]++;
    }
    int cnt = 0;
    for (int i = 1; i <= ccno; i++) if (in[i] == 1) cnt++;
    printf("%d\n", (cnt + 1) / 2);
}
