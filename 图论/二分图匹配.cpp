int mch[maxn], vis[maxn];
std::vector<int> e[maxn];
bool dfs(const int u, const int tag) {
    for (auto v : e[u]) {
        if (vis[v] == tag) continue;
        vis[v] = tag;
        if (!mch[v] || dfs(mch[v], tag)) return mch[v] = u, 1;
    }
    return 0;
}
int main() {
    int ans = 0;
    for (int i = 1; i <= n; ++i) if (dfs(i, i)) ++ans;
}