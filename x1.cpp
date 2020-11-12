#include <bits/stdc++.h>
using namespace std;

void solve(int n) 
{ 
    int k, x = 0; 
      
    int res[100];  
  
    for (k = 9; k > 1; k--) 
    { 
        while (n % k == 0) 
        { 
            n = n / k; 
            res[x] = k; 
            x++; 
        } 
    } 
  
    if (n > 10) 
    { 
        cout << "Not possible" << endl; 
        return; 
    } 
  
    for (k = x - 1; k >= 0; k--) 
        cout << res[k]; 
    cout << endl;    
} 

int main() {
    int a;
    cin >> a;

    if (a < 10) { 
        cout << a + 10; 
        return 0; 
    } 

   solve(a);
}