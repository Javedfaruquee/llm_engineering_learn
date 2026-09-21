#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(long long iterations, double param1, double param2) {
    double result = 1.0;
    // Loop unrolling and vectorization hint via OpenMP or manual restructuring if needed,
    // but a simple loop with -O3 -march=native and -ffast-math auto-vectorizes very well.
    for (long long i = 1; i <= iterations; ++i) {
        double fi = static_cast<double>(i);
        double j1 = fi * param1 - param2;
        double j2 = fi * param1 + param2;
        result -= (1.0 / j1);
        result += (1.0 / j2);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    double res = calculate(200_000_000, 4.0, 1.0) * 4.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Result: " << std::fixed << std::setprecision(12) << res << "\n";
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds\n";

    return 0;
}
