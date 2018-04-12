#include <chrono>


class ProgramTimer {
public:
	void Start() {
		startTimer = std::chrono::steady_clock::now();
	};

	long long int End() {
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimer).count();
	};
private:
	std::chrono::steady_clock::time_point startTimer;
};