#include <bits/stdc++.h>
using namespace std;

#define MOD 1000000007

int main() {
    int T;
    cin >> T;
    vector<long long int> fib(46, 1);
    for(int i=3;i<=45;++i) {
        fib[i] = (fib[i-1] + fib[i-2])%MOD;
    }
    while(T --) {
        int n;
        cin >> n;
        if(n < 3) {
            cout << 1 << endl;
        }
        else {
            long long int temp = ((fib[2*n - 5]%MOD)*(fib[2*n - 3]%MOD))%MOD;
            cout << temp << endl;
        }
    }
}