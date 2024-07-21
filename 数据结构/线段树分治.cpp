#include <bits/stdc++.h>
using namespace std;

const int N = 4e5 + 5, M = 4e6;
int tot, n, m, t, fa[N], d[N], u[N], v[N];
int stk[N], top, head[N];
bool fl[N];

struct E {
	int nxt, id;
} e[M];

int find(int x) {
	while (fa[x]) x = fa[x];
	return x;
}
void merge(int x, int y) {
    x = find(x), y = find(y);
    if (x == y) return;
	if (d[x] > d[y]) swap(x, y);
	fa[x] = y;
	stk[++top] = x;
    d[y] += fl[top] = (d[x] == d[y]);
}

void upd(int p, int l, int r, int L, int R, const int &i) {
    if (l == L && r == R) {
    	e[++tot] = (E){head[p], i};
    	head[p] = tot;
    	return ;
    }
    int mid = (l + r) >> 1;
    if (R <= mid) upd(p << 1, l, mid, L, R, i);
    else if (L > mid) upd(p << 1 | 1, mid + 1, r, L, R, i);
    else upd(p << 1, l, mid, L, mid, i), upd(p << 1 | 1, mid + 1, r, mid + 1, R, i);
}
void solve(int p, int l, int r) {
    int lst = top, mid = (l + r) >> 1;;
    for (int i = head[p]; i; i = e[i].nxt) {
    	int x = u[e[i].id], y = v[e[i].id];
    	if (find(x) == find(y)) {
    		for (int i = l; i <= r; i++) puts("No");
            goto che;
    	}
    	merge(x + n, y), merge(x, y + n);
    }
    if (l == r)
    	puts("Yes");
    else {
        solve(p << 1, l, mid);
        solve(p << 1 | 1, mid + 1, r);
    }
    che : for (; top > lst; top--) d[fa[stk[top]]] -= fl[top], fa[stk[top]] = 0;
}


int main() {
	//freopen("2.in", "r", stdin);
    scanf("%d %d %d", &n, &m, &t);
    for (int i = 1; i <= m; i++) {
    	int l, r;
    	scanf("%d %d %d %d", &u[i], &v[i], &l, &r);
        if (l == r) continue;
        upd(1, 1, t, l + 1, r, i);
    }
    solve(1, 0, t - 1);
    return 0;
}

