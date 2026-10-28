#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <numeric>

using namespace std;

// ---- CONFIG ----
const int NUM_HOUSES = 15;
const int PSO_PARTICLES = 30;
const int MAX_ITERATIONS = 100;
const double W = 0.8;
const double C1 = 2.0;
const double C2 = 2.0;
const double V_MAX = 2.0;

// ---- DATA STRUCTURES ----
struct Point { double x, y; };

struct House {
    int id;
    Point location;
    int priority;
    double time_window_end;
};

struct Particle {
    vector<double> position, velocity, pbest_position;
    double pbest_fitness;

    Particle() {
        position.resize(NUM_HOUSES);
        velocity.resize(NUM_HOUSES);
        pbest_position.resize(NUM_HOUSES);
        pbest_fitness = numeric_limits<double>::max();
    }
};

// ---- GLOBALS ----
Point post_office;
vector<House> all_houses;
vector<vector<double>> dist_matrix;
mt19937 gen(random_device{}());

double uniform_rand(double lo, double hi) {
    return uniform_real_distribution<>(lo, hi)(gen);
}

double euclidean(Point a, Point b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

void precompute_distances() {
    int N = NUM_HOUSES;
    dist_matrix.assign(N+1, vector<double>(N+1, 0.0));

    for (int i = 0; i < N; ++i) {
        double d = euclidean(post_office, all_houses[i].location);
        dist_matrix[0][i+1] = dist_matrix[i+1][0] = d;
    }

    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {
            double d = euclidean(all_houses[i].location, all_houses[j].location);
            dist_matrix[i+1][j+1] = dist_matrix[j+1][i+1] = d;
        }
}

void create_dummy_data() {
    post_office = {50, 50};
    all_houses = {
        {0,{10,20},1,9.0},{1,{85,15},2,12.0},{2,{30,75},1,8.5},
        {3,{65,80},3,16.0},{4,{20,50},2,11.0},{5,{75,35},1,9.5},
        {6,{40,10},3,17.0},{7,{55,65},2,13.0},{8,{15,90},1,10.0},
        {9,{90,60},2,14.0},{10,{35,40},3,15.0},{11,{70,90},1,8.0},
        {12,{25,25},2,12.5},{13,{80,50},3,16.5},{14,{50,85},2,11.5}
    };
}

// ---- SPV DECODE ----
vector<int> decode_position(const vector<double>& pos) {
    vector<pair<double,int>> ranked;
    for (int i = 0; i < NUM_HOUSES; ++i)
        ranked.push_back({pos[i], i});

    sort(ranked.begin(), ranked.end());

    vector<int> order;
    for (auto &p : ranked) order.push_back(p.second);
    return order;
}

// ---- FITNESS ----
double calculate_fitness(const vector<double>& pos) {
    vector<int> route = decode_position(pos);

    double total_dist = 0.0, penalty = 0.0;
    int cur = 0;
    double cur_time = 0.0;

    for (int h : route) {
        double d = dist_matrix[cur][h+1];
        total_dist += d;
        cur_time += d;

        if (cur_time > all_houses[h].time_window_end)
            penalty += (cur_time - all_houses[h].time_window_end) * 50.0;

        cur = h+1;
    }

    total_dist += dist_matrix[cur][0];
    return total_dist + penalty;
}

// ---- SOLVER ----
void solve_pso_standard() {
    vector<Particle> swarm(PSO_PARTICLES);
    vector<double> gbest_pos(NUM_HOUSES);
    double gbest_fit = numeric_limits<double>::max();

    for (auto &p : swarm) {
        for (int i = 0; i < NUM_HOUSES; ++i) {
            p.position[i] = uniform_rand(-2,2);
            p.velocity[i] = uniform_rand(-0.5,0.5);
        }

        p.pbest_fitness = calculate_fitness(p.position);
        p.pbest_position = p.position;

        if (p.pbest_fitness < gbest_fit) {
            gbest_fit = p.pbest_fitness;
            gbest_pos = p.pbest_position;
        }
    }

    cout << "-- Standard PSO --\n";

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        for (auto &p : swarm) {
            double fit = calculate_fitness(p.position);

            if (fit < p.pbest_fitness) {
                p.pbest_fitness = fit;
                p.pbest_position = p.position;
            }

            if (fit < gbest_fit) {
                gbest_fit = fit;
                gbest_pos = p.position;
            }
        }

        for (auto &p : swarm) {
            for (int i = 0; i < NUM_HOUSES; ++i) {
                double r1 = uniform_rand(0,1);
                double r2 = uniform_rand(0,1);

                p.velocity[i] = W*p.velocity[i]
                              + C1*r1*(p.pbest_position[i]-p.position[i])
                              + C2*r2*(gbest_pos[i]-p.position[i]);

                p.velocity[i] = max(-V_MAX, min(V_MAX, p.velocity[i]));
                p.position[i] += p.velocity[i];
            }
        }

        cout << "Iter " << iter+1 << " Best: " << gbest_fit << endl;
    }
}

int main() {
    create_dummy_data();
    precompute_distances();
    solve_pso_standard();
}