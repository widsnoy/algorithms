int main() {
    for (int i = 1; i <= n; i++) {
        v.push_back({i, a[i]});
        for (int j = (int)(v.size()) - 2; j >= 0; j--) {
            v[j].second = gcd(v[j].second, a[i]);
            if (v[j].second == v[j + 1].second) v.erase(v.begin() + j + 1);
        }
        mp[v[(int)(v.size()) - 1].second] += i - v[(int)(v.size()) - 1].first + 1;
        for (int j = (int)(v.size()) - 2; j >= 0; j--) {
            mp[v[j].second] += v[j + 1].first - v[j].first;
        }
    }
}