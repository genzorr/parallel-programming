#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <tuple>
#include "omp.h"

static long int q = 3;
static long int n = (long int)pow(2, (int)q)+1;
static long double d = 1.2;
static long double x_0 = -10, x_1 = 10;
static long double step = (x_1-x_0)/(long double)(n-1);
static long double y_0 = sqrt(2);
static long double y_1 = sqrt(2);

template<typename T>
T f(T y) {
    return 1000000*(y*y*y-y);
}

template<typename T>
std::vector<T>& F_grid(std::vector<T> &grid) {
    static std::vector<T> F_grid(n);
    F_grid[0] = step*step*f(y_0);
    for (long int i = 1; i < n; ++i)
        F_grid[i] = step*step*(f(grid[i]) + ((long double)1/12)*(f(grid[i+1]) - 2*f(grid[i]) + f(grid[i-1])));
    F_grid.back() = step*step*f(y_1);
    return F_grid;
}

//Vector

//template<typename T>
/*std::vector<T>& f(std::vector<T> &grid) {
    static std::vector<T> f_grid(n);
    for (long int i = 0; i < n; ++i)
        f_grid[i] = f(grid[i]);
    return f_grid;
}*/

typedef std::vector<long double> Vector;
typedef std::vector<std::vector<long double>> Matrix;

void print_vec(const Vector & vec){
    for (auto i: vec)
        printf("%.12Lf ", i);
    printf("\n");
}

double F_f(const Vector& grid, int i) {
    return (f(grid[i]) + ((long double)1/12)*(f(grid[i+1]) - 2*f(grid[i]) + f(grid[i-1])));
}


void print_mat(const Matrix& mat) {
    for (auto& s : mat)
        print_vec(s);
}

void init_sol(Vector& y, const Vector &x){
    y[0] = y_0;
    for (int i = 1; i < y.size()-1; ++i){
        y[i] = x[i];
    }
    y[n-1] = y_1;
}

Vector & getgrid(double func(double)){
    static Vector grid_vec(n);
    for (int i=0; i < n; ++i){
        grid_vec[i] = x_0 + step*i;
    }
    return grid_vec;
}

long double epsilon(const Vector& y, const Vector& F){
    long double e = 0;
    for (int i = 1; i < n-1; ++i){
        e += ((y[i+1]-2*y[i]+y[i-1])/step*step - F_f(y, i))*((y[i+1]-2*y[i]+y[i-1])/step*step - F_f(y, i));
    }
    return sqrt(e/n);
}

std::tuple<Vector, Vector, Vector> reduction_forward(Vector& F, long double a, long double b, long double c){
    static Vector A(q);
    static Vector B(q);
    static Vector C(q);
    A[0] = a;
    B[0] = b;
    C[0] = c;
    long int size = n-1;
    long int k;
    long int stp = 1;
    long int start = 2;
    long  double al, bet;
    //print_vec(F);
    for (long int i=0; i < q-1; ++i){
        al = -A[i]/B[i];
        bet = -C[i]/B[i];
        A[i+1] = al*A[i];
        B[i+1] = B[i] + 2*al*C[i];
        C[i+1] = bet*C[i];
        size = (size-1)/2;
#pragma omp parallel for private(k) firstprivate(start, al, bet) shared(F)
        for (long int j = 0; j < size; ++j){
            k = start*(j+1);
            F[k] = al*F[k-stp] + F[k] + bet*F[k+stp];
        }
        start *= 2;
        stp *= 2;
    }
    std::tuple<Vector, Vector, Vector> res = {A, B, C};
    return res;
}

Vector reduction_backward(const Vector& x, const Vector& a, const Vector& F, const Vector& b, const Vector& c){
    Vector result = x;
    long int start = (n-1)/2;
    long int stp = start;
    long int size = 1;
    long int k;
    long double al, bet;
    for (long int i = q-1; i >= 0; --i){
        al = -a[i]/b[i];
        bet = -c[i]/b[i];
#pragma omp parallel for private(k) firstprivate(start, al, bet) shared(F)
        for (long int j = 0; j < size; ++j){
            k = start*(2*j+1);
            result[k] = F[k]/b[i] + al*result[k-stp] + bet*result[k+stp];
        }
        start /= 2;
        stp = start;
        size *= 2;
    }
    return result;
}

int main(int argc, char** argv){
    omp_set_num_threads(1);
    q = (long int)std::strtol(argv[1], nullptr, 10);
    n = (long int)pow(2, (int)q)+1;
    step = (x_1-x_0)/(n-1);
    auto grid = getgrid(f);
    Vector y(n);
    init_sol(y, grid);
    auto F = F_grid(y);
    auto F_next = F_grid(y);

    long int size = n;
    Vector a, b, c;
    long double strt = omp_get_wtime();
    std::tie(a, b, c) = reduction_forward(F, 1, -2, 1);
    y = reduction_backward(y, a, F, b, c);
    //print_vec(y);
    long double stp = omp_get_wtime();
    std::ofstream fout;
    fout.open("result.txt");
    for(auto i : grid)
        fout << i << " ";
    fout << std::endl;
    for(auto i : y)
        fout << i << " ";
    fout.close();
    std::cout << stp-strt << std::endl;
    return 0;
}