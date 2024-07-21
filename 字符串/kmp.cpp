int n = strlen(s + 1);
for (int i = 2; i <= n; i++) {
    int j = k[i - 1];
    while (j != 0 && s[i] != s[j + 1]) j = k[j];
    if (s[i] == s[j + 1]) k[i] = j + 1;
    else k[i] = 0;
}
