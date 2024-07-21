int bccno[N], bcc_cnt, siz_e[N], siz_p[N], dfs_clock, low[N], dfn[N], top;
pair<int, int> stk[N];
void dfs(int u, int fa) {
    low[u] = dfn[u] = ++dfs_clock;
    for(int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v;
        if(v == fa) continue;
        if(!dfn[v]) {
            stk[++top] = make_pair(u, v);
            dfs(v, u);
            low[u] = min(low[u], low[v]);
            if(low[v] >= dfn[u]) {
                bcc_cnt++;
                while(true) {
                    int x = stk[top].first, y = stk[top].second;
                    top--;
                    siz_e[bcc_cnt]++;
                    if(bccno[x] != bcc_cnt) {bccno[x] = bcc_cnt; siz_p[bcc_cnt]++;}
                    if(bccno[y] != bcc_cnt) {bccno[y] = bcc_cnt; siz_p[bcc_cnt]++;}
                    if(x == u && y == v) break;
                }
            }
        } else if(dfn[v] < dfn[u]) {stk[++top] = make_pair(u, v); low[u] = min(low[u], dfn[v]);}
    }
}
