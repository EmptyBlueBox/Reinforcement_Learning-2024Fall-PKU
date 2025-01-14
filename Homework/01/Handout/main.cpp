#include <ctime>
#include <iostream>
#include <vector>

#include "tictactoe.hpp"

using namespace std;

class TicTacToePolicyBase{
    public:
        virtual TicTacToe::Action operator()(const TicTacToe::State& state) const = 0;
};

// randomly select a valid action for the step.
class TicTacToePolicyRandom : public TicTacToePolicyBase{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            int n_action = actions.size();
            int action_id = rand() % n_action;
            if (state.turn == TicTacToe::PLAYER_X){
                return actions[action_id];
            } else {
                return actions[action_id];
            }
        }
        TicTacToePolicyRandom(){
            srand(time(nullptr));
        }
};

// select the first valid action.
class TicTacToePolicyDefault : public TicTacToePolicyBase{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            if (state.turn == TicTacToe::PLAYER_X){
                // TODO
                
                return actions[0];
            } else {
                return actions[0];
            }
        }
        TicTacToePolicyDefault(){}
};


#include <chrono>
#include <thread>

// randomly select action
int main(){
    bool done = false;
    // set verbose true
    TicTacToe env(true);
    // TicTacToePolicyDefault policy;
    TicTacToePolicyRandom policy;
    while (not done){
        TicTacToe::State state = env.get_state();
        TicTacToe::Action action = policy(state);
        env.step(action);
        done = env.done();
        // env.step_back();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    int winner = env.winner();
    return 0;
};