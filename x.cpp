#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    cin >> N;
    if(N<2) {
        cout << "Invalid Input" << endl;
    }
    int a[N];
    for(int i=0;i<N;++i) {
        cin >> a[i];
    }
    sort(a, a+N);
    int flag = 0;
    for(int i=0;i<N-1;++i) {
        if(a[i] != a[i+1]){
            flag = 1;
            break;
        }
    }
    if(flag == 1) {
        cout << a[0] << " " << a[1] << endl;
    }
    else {
        cout << "Equal" << endl;
    }
}