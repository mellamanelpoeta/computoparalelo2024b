#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <omp.h>
#include <queue>
#include <atomic>
#include <thread> // For std::this_thread::yield()

struct Point {
    double x, y;
};

using PointCloud = std::vector<Point>;

double euclideanDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

void findNeighbors(const PointCloud& points, std::vector<std::vector<int>>& neighbors,
                   std::vector<int>& neighborsNumber,
                   double eps, int start_value, int step) {
    for (int i = start_value; i < points.size(); i += step) {
        for (int j = i + 1; j < points.size(); j++) {
            if (euclideanDistance(points[i], points[j]) <= eps) {
                #pragma omp critical
                {
                    neighbors[i].push_back(j);
                    neighbors[j].push_back(i);
                }
            }
        }
        neighborsNumber[i] = neighbors[i].size();
    }
}

void serialAnalyzePoints(const PointCloud& points, const std::vector<std::vector<int>>& neighbors,
                         const std::vector<int>& neighborsNumber, int minPts, std::vector<int>& clusters) {
    int currentCluster = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (clusters[i] != -1) continue;

        if (neighborsNumber[i] < minPts) {
            clusters[i] = -1;  // Noise point
            continue;
        }

        std::queue<int> neighborQueue;
        clusters[i] = currentCluster;
        neighborQueue.push(i);

        while (!neighborQueue.empty()) {
            int currentPoint = neighborQueue.front();
            neighborQueue.pop();

            for (int neighbor : neighbors[currentPoint]) {
                if (clusters[neighbor] == -1) {
                    clusters[neighbor] = currentCluster;
                    if (neighborsNumber[neighbor] >= minPts) {
                        neighborQueue.push(neighbor);
                    }
                }
            }
        }

        currentCluster++;
    }
}

void parallelDBSCAN(PointCloud& points, std::vector<int>& clusters,
                    double eps, int minPts, int findNeighboursThreadsNum) {
    int totalThreads = findNeighboursThreadsNum;
    omp_set_num_threads(totalThreads);

    std::vector<std::vector<int>> neighbors(points.size());
    std::vector<int> neighborsNumber(points.size(), 0);

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        findNeighbors(points, neighbors, neighborsNumber, eps, threadId, findNeighboursThreadsNum);
    }

    // Perform cluster assignment serially
    serialAnalyzePoints(points, neighbors, neighborsNumber, minPts, clusters);
}

PointCloud readCSV(const std::string& filename) {
    PointCloud cloud;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open the file: " + filename);
    }

    std::string line;
    int lineCount = 0;
    while (std::getline(file, line)) {
        lineCount++;
        std::istringstream iss(line);
        std::string x_str, y_str;

        if (std::getline(iss, x_str, ',') && std::getline(iss, y_str, ',')) {
            try {
                Point p;
                p.x = std::stod(x_str);
                p.y = std::stod(y_str);
                cloud.push_back(p);
            } catch (const std::exception& e) {
                std::cerr << "Error converting values on line " << lineCount << ": " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Incorrect format on line " << lineCount << std::endl;
        }
    }

    return cloud;
}

void writeCSV(const std::string& filename, const PointCloud& cloud, const std::vector<int>& clusters) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open the file for writing: " + filename);
    }

    for (size_t i = 0; i < cloud.size(); ++i) {
        const auto& point = cloud[i];
        int clusterId = clusters[i];
        file << point.x << "," << point.y << "," << clusterId << "\n";
    }

    if (file.fail()) {
        throw std::runtime_error("An error occurred while writing to the file: " + filename);
    }

    file.close();
}

template<typename Func, typename... Args>
double measureExecutionTime(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file.csv> <output_file.csv> <epsilon> <min_points> <find_neighbours_threads>\n";
        return 1;
    }

    try {
        std::string input_file = argv[1];
        std::string output_file = argv[2];
        double eps = std::stod(argv[3]);
        int minPts = std::stoi(argv[4]);
        int findNeighboursThreadsNum = std::stoi(argv[5]);

        PointCloud cloud = readCSV(input_file);

        // Initialize clusters to -1 (unclassified)
        std::vector<int> clusters(cloud.size(), -1);

        double time = measureExecutionTime(parallelDBSCAN, std::ref(cloud), std::ref(clusters), eps, minPts, findNeighboursThreadsNum);
        std::cout << "Parallel DBSCAN execution time: " << time << " ms\n";

        // Count clusters and noise points
        int numClusters = 0;
        int noisePoints = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            int clusterId = clusters[i];
            if (clusterId == -1) {
                noisePoints++;
            } else {
                numClusters = std::max(numClusters, clusterId + 1);
            }
        }

        std::cout << "Number of clusters found: " << numClusters << std::endl;
        std::cout << "Number of noise points: " << noisePoints << std::endl;

        writeCSV(output_file, cloud, clusters);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
