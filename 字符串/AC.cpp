namespace AC {
	int ch[N][26], tot, fail[N], e[N];
	void insert(const char *s) {
		int u = 0, n = strlen(s + 1);
		for (int i = 1; i <= n; i++) {
            if (!ch[u][s[i] - 'a']) ch[u][s[i] - 'a'] = ++tot;
            u = ch[u][s[i] - 'a'];
		}
		e[u] += 1;
	}
	void build() {
		queue<int> q;
		for (int i = 0; i <= 25; i++) if (ch[0][i]) q.push(ch[0][i]);
		while (!q.empty()) {
			int now = q.front(); q.pop();
			for (int i = 0; i < 26; i++) {
				if (ch[now][i]) fail[ch[now][i]] = ch[fail[now]][i], q.push(ch[now][i]);
				else ch[now][i] = ch[fail[now]][i];
			}
		}
	}
	int query(const char *s) {
        int u = 0, n = strlen(s + 1), res = 0;
        for (int i = 1; i <= n; i++){
        	u = ch[u][s[i] - 'a'];
        	for (int j = u; j && e[j] != -1; j = fail[j]) {
        		res += e[j];
        		e[j] = -1;
        	}
        }
        return res;
	}
}
