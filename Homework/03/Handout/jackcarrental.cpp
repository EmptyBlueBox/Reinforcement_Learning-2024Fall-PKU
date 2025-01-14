#include <ctime>
#include <random>
#include <utility>
#include <iostream>
#include <algorithm>

using namespace std;

class JackCarRental{
    static const int
        MAX_CAR_1,
        MAX_CAR_2,
        MOVE_LIMIT;
    static const double
        MOVE_COST,
        RENT_PRICE,
        MEAN_REQUEST_1,
        MEAN_REQUEST_2,
        MEAN_RETURN_1,
        MEAN_RETURN_2;
    static poisson_distribution<int>
        request_1, request_2, return_1, return_2;
    public:
        typedef pair<int, int> State;
        bool verbose;
        State state(){
            return make_pair(car_1, car_2);
        }
        void set_state(int car_1, int car_2){
            if (verbose){
                cout << "State set to (" << car_1 << ", " << car_2 << ")" << endl;
            }
            this->car_1 = car_1;
            this->car_2 = car_2;
        }
        void reset(){
            if (verbose){
                cout << "Environment reset." << endl;
            }
            day = 0;
            car_1 = 0;
            car_2 = 0;
        }
        pair<State, double> step(int action){
            day ++;
            if (verbose){
                cout << "\nDay: " << day 
                    << " State: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            double reward = state_transition(action);
            if (verbose){
                cout << "\tReward: " << reward << endl;
            }
            return make_pair(state(), reward);
        }
        int sample_action(){
            int action_low = max(-car_2, -MOVE_LIMIT);
            int action_high = min(car_1, MOVE_LIMIT);
            uniform_int_distribution<int> random_action(action_low, action_high);
            int action = random_action(e);
            if (verbose){
                cout << "\tAction " << action 
                    << " sampled from uniform[" << action_low << ", " << action_high << "]" << endl;
            }
            return action;
        }
        JackCarRental(int car_1=0, int car_2=0, bool verbose=false){
            this->day = 0;
            this->verbose = verbose;
            set_state(car_1, car_2);
            e.seed(time(nullptr));
        }
    private:
        int car_1, car_2, day;
        default_random_engine e;
        
        double state_transition(int action){
            car_1 = min(car_1 - action, MAX_CAR_1);
            car_2 = min(car_2 + action, MAX_CAR_2);
            double total_move_cost = abs(action) * MOVE_COST;
            if (verbose){
                cout << "\tMove: (" << -action << ", " << action 
                    << "), cost: " << total_move_cost << endl; 
                cout << "\tAfter movement, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            int req_1 = this->request_1(e);
            int req_2 = request_2(e);
            if (verbose){
                cout << "\tRental request: (" << req_1 << ", " << req_2 << ")" << endl; 
            } 
            int rent_1 = min(car_1, req_1);
            int rent_2 = min(car_2, req_2);
            double total_income = (rent_1 + rent_2) * RENT_PRICE;
            car_1 -= rent_1;
            car_2 -= rent_2;
            if (verbose){
                cout << "\tRent: (" << rent_1 << ", " << rent_2 
                    << "), income: " << total_income << endl;
                cout << "\tAfter rent, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            int ret_1 = return_1(e);
            int ret_2 = return_2(e);
            if (verbose){
                cout << "\tCars to return: (" << ret_1 << ", " << ret_2 << ")" << endl;
            }
            car_1 = min(car_1 + ret_1, MAX_CAR_1);
            car_2 = min(car_2 + ret_2, MAX_CAR_2);
            if (verbose){
                cout << "\tAfter return, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            return total_income - total_move_cost;
        }
};

const int
    JackCarRental::MAX_CAR_1 = 20,
    JackCarRental::MAX_CAR_2 = 20,
    JackCarRental::MOVE_LIMIT = 5;
const double 
    JackCarRental::MOVE_COST = 2.0,
    JackCarRental::RENT_PRICE = 10.0,
    JackCarRental::MEAN_REQUEST_1 = 3.0,
    JackCarRental::MEAN_REQUEST_2 = 4.0,
    JackCarRental::MEAN_RETURN_1 = 3.0,
    JackCarRental::MEAN_RETURN_2 = 2.0;
poisson_distribution<int> 
    JackCarRental::request_1(JackCarRental::MEAN_REQUEST_1),
    JackCarRental::request_2(JackCarRental::MEAN_REQUEST_2),
    JackCarRental::return_1(JackCarRental::MEAN_RETURN_1),
    JackCarRental::return_2(JackCarRental::MEAN_RETURN_2);

#include <chrono>
#include <thread>
int main(){
    JackCarRental env(0, 0, true);
    while (true){
        int action = env.sample_action();
        env.step(action);
        this_thread::sleep_for(chrono::milliseconds(1000));
    }
    return 0;
}