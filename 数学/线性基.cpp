struct LinerBasis {
    int a[20], pos[20];
    void add(int v, int p) {
        for (int i = 19; i >= 0; i--) if ((v >> i) & 1) {
            if (a[i]) {
                if (p > pos[i]) {
                    swap(p, pos[i]);
                    swap(a[i], v);
                }
                v ^= a[i];
            } else {
                a[i] = v;
                pos[i] = p;
                return;
            }
        }
    }
} b[N];

LinerBasis operator + (LinerBasis a, LinerBasis b) {
    for (int i = 19; i >= 0; i--) {
        if (b.a[i]) a.add(b.a[i], b.pos[i]);
    }
    return a;
}