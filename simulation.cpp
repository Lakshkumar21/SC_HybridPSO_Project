#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <iomanip>

using namespace std;

// ---------------- CONFIG ----------------
const int NUM_HOUSES = 15;
const int PSO_PARTICLES = 30;
const int MAX_ITERATIONS = 100;

// Standard PSO params
const double W1 = 0.8;
const double C1_1 = 2.0;
const double C2_1 = 2.0;
const double V_MAX1 = 2.0;

// Hybrid PSO params
const double W2 = 0.729;
const double C1_2 = 1.49445;
const double C2_2 = 1.49445;
const double V_MAX2 = 0.3;

// ---------------- DATA ----------------
struct Point { double x, y; };

struct House {
    int id;
    Point location;
    int priority;
    double time_window_end;
};

Point post_office;
vector<House> all_houses;
vector<vector<double>> dist_matrix;

mt19937 gen(random_device{}());

double uniform_rand(double lo, double hi) {
    return uniform_real_distribution<>(lo, hi)(gen);
}

double euclidean(Point a, Point b) {
    return sqrt((a.x-a.x)*(a.x-a.x) + (a.y-b.y)*(a.y-b.y));
}

// ---------------- INIT ----------------
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

void precompute_distances() {
    int N = NUM_HOUSES;
    dist_matrix.assign(N+1, vector<double>(N+1, 0));

    for(int i=0;i<N;i++){
        double d = euclidean(post_office, all_houses[i].location);
        dist_matrix[0][i+1] = dist_matrix[i+1][0] = d;
    }

    for(int i=0;i<N;i++)
        for(int j=i;j<N;j++){
            double d = euclidean(all_houses[i].location, all_houses[j].location);
            dist_matrix[i+1][j+1] = dist_matrix[j+1][i+1] = d;
        }
}

// ---------------- STANDARD PSO ----------------
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

vector<int> decode_position(const vector<double>& pos) {
    vector<pair<double,int>> ranked;
    for(int i=0;i<NUM_HOUSES;i++)
        ranked.push_back({pos[i], i});

    sort(ranked.begin(), ranked.end());

    vector<int> order;
    for(auto &p: ranked) order.push_back(p.second);
    return order;
}

double calculate_fitness(const vector<double>& pos) {
    vector<int> route = decode_position(pos);

    double total = 0, penalty = 0;
    int cur = 0;
    double time = 0;

    for(int h: route){
        double d = dist_matrix[cur][h+1];
        total += d;
        time += d;

        if(time > all_houses[h].time_window_end)
            penalty += (time - all_houses[h].time_window_end)*50;

        cur = h+1;
    }

    total += dist_matrix[cur][0];
    return total + penalty;
}

void solve_pso_standard() {
    vector<Particle> swarm(PSO_PARTICLES);
    vector<double> gbest_pos(NUM_HOUSES);
    double gbest = 1e18;

    for(auto &p: swarm){
        for(int i=0;i<NUM_HOUSES;i++){
            p.position[i] = uniform_rand(-2,2);
            p.velocity[i] = uniform_rand(-0.5,0.5);
        }

        p.pbest_fitness = calculate_fitness(p.position);
        p.pbest_position = p.position;

        if(p.pbest_fitness < gbest){
            gbest = p.pbest_fitness;
            gbest_pos = p.position;
        }
    }

    cout << "\n--- Standard PSO ---\n";

    for(int iter=0;iter<MAX_ITERATIONS;iter++){
        for(auto &p: swarm){
            double fit = calculate_fitness(p.position);

            if(fit < p.pbest_fitness){
                p.pbest_fitness = fit;
                p.pbest_position = p.position;
            }

            if(fit < gbest){
                gbest = fit;
                gbest_pos = p.position;
            }
        }

        for(auto &p: swarm){
            for(int i=0;i<NUM_HOUSES;i++){
                double r1 = uniform_rand(0,1);
                double r2 = uniform_rand(0,1);

                p.velocity[i] = W1*p.velocity[i]
                              + C1_1*r1*(p.pbest_position[i]-p.position[i])
                              + C2_1*r2*(gbest_pos[i]-p.position[i]);

                p.velocity[i] = max(-V_MAX1, min(V_MAX1, p.velocity[i]));
                p.position[i] += p.velocity[i];
            }
        }
    }

    vector<int> route = decode_position(gbest_pos);

    cout << "\nFinal Best Route (Standard PSO):\n";
    cout << "PostOffice -> ";
    for(int x: route) cout << x << " -> ";
    cout << "PostOffice\n";
}

// ---------------- HYBRID PSO ----------------
double run_greedy(double w_prox) {
    double w_pri = 1 - w_prox;

    vector<int> unvisited;
    for(int i=0;i<NUM_HOUSES;i++) unvisited.push_back(i);

    int cur = 0;
    double time=0, total=0, penalty=0;

    while(!unvisited.empty()){
        double best = 1e18;
        int best_i=-1, best_h=-1;

        for(int i=0;i<unvisited.size();i++){
            int h = unvisited[i];
            double d = dist_matrix[cur][h+1];
            double arrive = time + d;
            double slack = all_houses[h].time_window_end - arrive;

            double score = w_prox*d - w_pri*slack/(double)all_houses[h].priority;

            if(score < best){
                best = score;
                best_i = i;
                best_h = h;
            }
        }

        double d = dist_matrix[cur][best_h+1];
        time += d;
        total += d;

        if(time > all_houses[best_h].time_window_end)
            penalty += (time - all_houses[best_h].time_window_end)*50;

        cur = best_h+1;
        unvisited.erase(unvisited.begin()+best_i);
    }

    total += dist_matrix[cur][0];
    return total + penalty;
}

vector<int> get_best_route(double w_prox) {
    double w_pri = 1 - w_prox;

    vector<int> unvisited, route;
    for(int i=0;i<NUM_HOUSES;i++) unvisited.push_back(i);

    int cur=0;
    double time=0;

    while(!unvisited.empty()){
        double best=1e18;
        int bi=-1, bh=-1;

        for(int i=0;i<unvisited.size();i++){
            int h=unvisited[i];
            double d = dist_matrix[cur][h+1];
            double arrive = time + d;
            double slack = all_houses[h].time_window_end - arrive;

            double score = w_prox*d - w_pri*slack/(double)all_houses[h].priority;

            if(score<best){
                best=score; bi=i; bh=h;
            }
        }

        double d = dist_matrix[cur][bh+1];
        time += d;
        cur = bh+1;
        route.push_back(bh);
        unvisited.erase(unvisited.begin()+bi);
    }

    return route;
}

void solve_pso_hybrid() {
    struct P{ double pos,vel,pbest_pos,pbest_fit=1e18; };

    vector<P> swarm(PSO_PARTICLES);
    double gbest=1e18, gpos=0.5;

    for(auto &p: swarm){
        p.pos = uniform_rand(0,1);
        p.vel = uniform_rand(-0.2,0.2);
        p.pbest_pos = p.pos;
    }

    cout << "\n--- Hybrid PSO ---\n";

    for(int iter=0;iter<MAX_ITERATIONS;iter++){
        for(auto &p: swarm){
            double fit = run_greedy(p.pos);

            if(fit < p.pbest_fit){
                p.pbest_fit = fit;
                p.pbest_pos = p.pos;
            }

            if(fit < gbest){
                gbest = fit;
                gpos = p.pos;
            }
        }

        for(auto &p: swarm){
            double r1=uniform_rand(0,1), r2=uniform_rand(0,1);

            p.vel = W2*p.vel + C1_2*r1*(p.pbest_pos-p.pos)
                             + C2_2*r2*(gpos-p.pos);

            p.vel = max(-V_MAX2, min(V_MAX2,p.vel));
            p.pos = max(0.0, min(1.0,p.pos+p.vel));
        }
    }

    vector<int> route = get_best_route(gpos);

    cout << "\nFinal Best Route (Hybrid PSO):\n";
    cout << "PostOffice -> ";
    for(int x: route) cout << x << " -> ";
    cout << "PostOffice\n";
}

// ---------------- MAIN ----------------
int main() {
    create_dummy_data();
    precompute_distances();

    solve_pso_standard();
    solve_pso_hybrid();

    return 0;
}