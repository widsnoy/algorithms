typedef unsigned long long ull;
int sqrt1(ull x) {
    ull y = sqrt(x);
    return y * y == x;
}
constexpr ull calc_table(int k) {
    ull table = 0;
    for (int i = 0; i < 64; i++) 
        table |= 1ull << (i * i % (1 << k));
    return table;
}
int sqrt4(ull x) {
    constexpr int k = 6;
    constexpr auto table = calc_table(k);
    ull y = x % (1 << k);
    if ((table >> y) & 1) return sqrt1(x);
    return 0;
}