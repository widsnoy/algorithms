#include <bits/stdc++.h>
//#include <ext/pb_ds/assoc_container.hpp>
//#include <ext/pb_ds/tree_policy.hpp>
//using namespace __gnu_pbds;
using namespace std;

#define fi first
#define se second
typedef pair<int, int> pii;
typedef long long ll;
typedef long double ld;
//std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());

const int mod = 998244353;
const int N = 202 + 5;

int n, m, s, t, s1, t1, head[N], cur[N], ecnt, d[N], vis[N];
int in[N], out[N];
struct Edge {
    int nxt, v, flow, cap;
}e[9999 * 5];
void add_edge(int u, int v, int flow, int cap) {
    e[ecnt] = {head[u], v, flow, cap}; head[u] = ecnt++;
    e[ecnt] = {head[v], u, flow, 0}; head[v] = ecnt++;
}
bool bfs() {
    memset(vis, 0, sizeof vis);
    std::queue<int> q;
    q.push(s);
    vis[s] = 1;
    d[s] = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int i = head[u]; i != -1; i = e[i].nxt) {
            int v = e[i].v;
            if (vis[v] || e[i].flow >= e[i].cap) continue;
            d[v] = d[u] + 1;
            vis[v] = 1;
            q.push(v);
        }
    }
    return vis[t];
}
int dfs(int u, int a) {
    if (u == t || !a) return a;
    int flow = 0, f;
    for (int& i = cur[u]; i != -1; i = e[i].nxt) {
        int v = e[i].v;
        if (d[u] + 1 == d[v] && (f = dfs(v, std::min(a, e[i].cap - e[i].flow))) > 0) {
            e[i].flow += f;
            e[i ^ 1].flow -= f;
            flow += f;
            a -= f;
            if (!a) break;
        }
    } 
    return flow;
}

void MAIN() {
    memset(head, -1, sizeof head);
    cin >> n >> m >> s1 >> t1;
    for (int i = 1; i <= m; i++) {
        int u, v, low, hig;
        cin >> u >> v >> low >> hig;
        in[v] += low; out[u] += low;
        add_edge(u, v, 0, hig - low);
    }
    add_edge(t1, s1, 0, 1000000000);
    s = 0, t = n + 1;
    for (int i = 1; i <= n; i++) {
        if (in[i] < out[i]) add_edge(i, t, 0, out[i] - in[i]);
        if (out[i] < in[i]) add_edge(s, i, 0, in[i] - out[i]);
    }
    while (bfs()) {
        for (int i = s; i <= t; i++) cur[i] = head[i];
        dfs(s, 1000000000);   
    }
    for (int i = head[s]; i != -1; i = e[i].nxt) {
        if (e[i].flow != e[i].cap) return cout << "please go home to sleep\n", void();
    }
    s = s1, t = t1;
    int ans = 0;
    for (int i = head[t]; i != -1; i = e[i].nxt) {
        int v = e[i].v;
        if (v == s) {
            ans = e[i].flow;
            e[i].flow = e[i ^ 1].flow = 0;
            e[i].cap = e[i ^ 1].cap = 0;
        }
    }
    while (bfs()) {
        for (int i = 0; i <= n + 1; i++) cur[i] = head[i];
        ans += dfs(s, 1000000000);   
    }
    cout << ans << '\n';
}

int main() {
    ios::sync_with_stdio(0), cin.tie(0);
    int T = 1;
  //  cin >> T;
    while (T--) MAIN();
    return 0;
} 
