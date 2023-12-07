const int N = 526010, MX = 22;
int ch[N * MX][2], tot, rt[N], w[N * MX], xorv[N * MX], val[N];
ll ans;

void pushup(int u) {
	w[u] = xorv[u] = 0;
	if (ch[u][0]) {
        w[u] += w[ch[u][0]];
        xorv[u] ^= (xorv[ch[u][0]] << 1);
	}
	if (ch[u][1]) {
	    w[u] += w[ch[u][1]];
	    xorv[u] ^= (xorv[ch[u][1]] << 1) | (w[ch[u][1]] & 1);    
    }
    w[u] &= 1;
}
void insert(int &o, ll ux, int dep) {
    if (!o) o = ++tot;
    if (dep > MX) return (void)(w[o]++);
    insert(ch[o][ux & 1], ux >> 1, dep + 1);
    pushup(o);
}
void addall(int o) {
	swap(ch[o][0], ch[o][1]);
	if (ch[o][0]) addall(ch[o][0]);
	pushup(o);
}
int merge(int a, int b) {
	if (!b || !a) return a + b;
	xorv[a] ^= xorv[b];
	w[a] += w[b];
	ch[a][0] = merge(ch[a][0], ch[b][0]);
	ch[a][1] = merge(ch[a][1], ch[b][1]);
	return a;
}

vector<int> G[N];
int read() {
	int w = 0, f = 1; char ch = getchar();
	while (ch > '9' || ch < '0') {
		if (ch == '-') f = -1;
		ch = getchar();
	}
	while (ch >= '0' && ch <= '9') {
		w = w * 10 + ch - 48;
		ch = getchar();
	}
	return w * f;
}

void dfs(int u) {
    for (auto v : G[u]) {
        dfs(v);
        rt[u] = merge(rt[u], rt[v]);
    }
    addall(rt[u]);
    insert(rt[u], val[u], 0);
    ans += (ll)xorv[rt[u]];
}

int main() {
    int n = read();
    for (int i = 1; i <= n; i++) val[i] = read();
    for (int i = 2; i <= n; i++) G[read()].push_back(i);
    dfs(1);
    printf("%lld\n", ans);
    return 0;
}
