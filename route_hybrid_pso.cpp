#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <iomanip>

using namespace std;

const int NUM_HOUSES = 15;
const int PSO_PARTICLES = 30;
const int MAX_ITERATIONS = 100;
const double W = 0.729;
const double C1 = 1.49445;
const double C2 = 1.49445;
const double V_MAX = 0.3;

struct Point { double x, y; };
struct House { int id; Point location; int priority; double time_window_end; };

Point post_office;
vector<House> all_houses;
vector<vector<double>> dist_matrix;
mt19937 gen(random_device{}());

double uniform_rand(double lo,double hi){
    return uniform_real_distribution<>(lo,hi)(gen);
}

double euclidean(Point a,Point b){
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

void precompute_distances(){
    int N=NUM_HOUSES;
    dist_matrix.assign(N+1,vector<double>(N+1,0));

    for(int i=0;i<N;i++){
        double d=euclidean(post_office,all_houses[i].location);
        dist_matrix[0][i+1]=dist_matrix[i+1][0]=d;
    }

    for(int i=0;i<N;i++)
        for(int j=i;j<N;j++){
            double d=euclidean(all_houses[i].location,all_houses[j].location);
            dist_matrix[i+1][j+1]=dist_matrix[j+1][i+1]=d;
        }
}

void create_dummy_data(){
    post_office={50,50};
    all_houses={
        {0,{10,20},1,9.0},{1,{85,15},2,12.0},{2,{30,75},1,8.5},
        {3,{65,80},3,16.0},{4,{20,50},2,11.0},{5,{75,35},1,9.5},
        {6,{40,10},3,17.0},{7,{55,65},2,13.0},{8,{15,90},1,10.0},
        {9,{90,60},2,14.0},{10,{35,40},3,15.0},{11,{70,90},1,8.0},
        {12,{25,25},2,12.5},{13,{80,50},3,16.5},{14,{50,85},2,11.5}
    };
}

// ---- GREEDY FITNESS ----
double run_greedy(double w_prox){
    double w_pri=1.0-w_prox;

    vector<int> unvisited;
    for(int i=0;i<NUM_HOUSES;i++) unvisited.push_back(i);

    int cur=0;
    double cur_time=0,total=0,penalty=0;

    while(!unvisited.empty()){
        double best=1e18;
        int best_i=-1,best_h=-1;

        for(int i=0;i<unvisited.size();i++){
            int h=unvisited[i];
            double d=dist_matrix[cur][h+1];
            double arrive=cur_time+d;
            double slack=all_houses[h].time_window_end-arrive;

            double score=w_prox*d - w_pri*slack/(double)all_houses[h].priority;

            if(score<best){
                best=score;
                best_i=i;
                best_h=h;
            }
        }

        double d=dist_matrix[cur][best_h+1];
        cur_time+=d;
        total+=d;

        if(cur_time>all_houses[best_h].time_window_end)
            penalty+=(cur_time-all_houses[best_h].time_window_end)*50;

        cur=best_h+1;
        unvisited.erase(unvisited.begin()+best_i);
    }

    total+=dist_matrix[cur][0];
    return total+penalty;
}

// ---- PSO ----
void solve_pso_hybrid(){
    struct Particle{ double pos,vel,pbest_pos,pbest_fit=1e18; };

    vector<Particle> swarm(PSO_PARTICLES);
    double gbest=1e18, gpos=0.5;

    for(auto &p:swarm){
        p.pos=uniform_rand(0,1);
        p.vel=uniform_rand(-0.2,0.2);
        p.pbest_pos=p.pos;
    }

    cout<<"-- Hybrid PSO --\n";

    for(int iter=0;iter<MAX_ITERATIONS;iter++){
        for(auto &p:swarm){
            double fit=run_greedy(p.pos);

            if(fit<p.pbest_fit){
                p.pbest_fit=fit;
                p.pbest_pos=p.pos;
            }

            if(fit<gbest){
                gbest=fit;
                gpos=p.pos;
            }
        }

        for(auto &p:swarm){
            double r1=uniform_rand(0,1),r2=uniform_rand(0,1);

            p.vel=W*p.vel + C1*r1*(p.pbest_pos-p.pos)
                         + C2*r2*(gpos-p.pos);

            p.vel=max(-V_MAX,min(V_MAX,p.vel));
            p.pos=max(0.0,min(1.0,p.pos+p.vel));
        }

        cout<<"Iter "<<iter+1<<" Best: "<<gbest<<endl;
    }
}

int main(){
    create_dummy_data();
    precompute_distances();
    solve_pso_hybrid();
}