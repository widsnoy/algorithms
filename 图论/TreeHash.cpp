const ull mask = chrono::steady_clock::now().time_since_epoch().count();

ull shift(ull x) {
    x ^= mask;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x ^= mask;
    return x;
}
int n;
ull H[N];
vector<int> G[N];
set<ull> s;

void dfs(int u, int fa) {
    H[u] = 1;
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs(v, u);
        H[u] += shift(H[v]);
    }
    s.emplace(H[u]);
} 
