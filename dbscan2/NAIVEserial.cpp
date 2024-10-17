#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <algorithm>

struct Point {
    double x, y;
    int cluster = -1;  // -1 represents an unclassified point
};

using PointCloud = std::vector<Point>;

double euclideanDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

std::vector<int> rangeQuery(const PointCloud& points, int p, double eps) {
    std::vector<int> neighbors;
    for (int i = 0; i < points.size(); i++) {
        if (euclideanDistance(points[p], points[i]) <= eps) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

void dbscan(PointCloud& points, double eps, int minPts) {
    int clusterIdx = 0;
    std::vector<bool> visited(points.size(), false);

    for (int i = 0; i < points.size(); i++) {
        if (visited[i]) continue;

        visited[i] = true;
        std::vector<int> neighbors = rangeQuery(points, i, eps);
        if (neighbors.size() < minPts) {
            points[i].cluster = -1;  // Mark as noise
        } else {
            points[i].cluster = clusterIdx;
            for (int j = 0; j < neighbors.size(); j++) {
                int neighborIdx = neighbors[j];
                if (!visited[neighborIdx]) {
                    visited[neighborIdx] = true;
                    std::vector<int> neighborNeighbors = rangeQuery(points, neighborIdx, eps);
                    if (neighborNeighbors.size() >= minPts) {
                        neighbors.insert(neighbors.end(), neighborNeighbors.begin(), neighborNeighbors.end());
                    }
                }
                if (points[neighborIdx].cluster == -1) {
                    points[neighborIdx].cluster = clusterIdx;
                }
            }
            clusterIdx++;
        }
    }
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

void writeCSV(const std::string& filename, const PointCloud& cloud) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open the file for writing: " + filename);
    }

    for (const auto& point : cloud) {
        file << point.x << "," << point.y << "," << point.cluster << "\n";
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
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input_file.csv> <output_file.csv> <epsilon> <min_points>\n";
        return 1;
    }

    try {
        std::string input_file = argv[1];
        std::string output_file = argv[2];
        double eps = std::stod(argv[3]);
        int minPts = std::stoi(argv[4]);

        PointCloud cloud = readCSV(input_file);

        double time = measureExecutionTime(dbscan, std::ref(cloud), eps, minPts);
        std::cout << "DBSCAN execution time: " << time << " ms\n";

        writeCSV(output_file, cloud);

        // Count clusters and noise points
        int numClusters = 0;
        int noisePoints = 0;
        for (const auto& point : cloud) {
            if (point.cluster == -1) {
                noisePoints++;
            } else {
                numClusters = std::max(numClusters, point.cluster + 1);
            }
        }

        std::cout << "Number of clusters found: " << numClusters << std::endl;
        std::cout << "Number of noise points: " << noisePoints << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}