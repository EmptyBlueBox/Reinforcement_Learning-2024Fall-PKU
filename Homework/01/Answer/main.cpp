#include <ctime>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>

#include "tictactoe.hpp"

using namespace std;

const int training_iterations = 100000; // Number of training games
bool verbose = false; // Set to true to see each game

class TicTacToePolicyBase
{
public:
    virtual TicTacToe::Action operator()(const TicTacToe::State &state) const = 0;
};

// randomly select a valid action for the step.
class TicTacToePolicyRandom : public TicTacToePolicyBase
{
public:
    TicTacToe::Action operator()(const TicTacToe::State &state) const
    {
        vector<TicTacToe::Action> actions = state.action_space();
        int n_action = actions.size();
        int action_id = rand() % n_action;
        if (state.turn == TicTacToe::PLAYER_X)
        {
            return actions[action_id];
        }
        else
        {
            return actions[action_id];
        }
    }
    TicTacToePolicyRandom()
    {
        srand(time(nullptr));
    }
};

// select the first valid action.
class TicTacToePolicyDefault : public TicTacToePolicyBase
{
public:
    TicTacToe::Action operator()(const TicTacToe::State &state) const
    {
        vector<TicTacToe::Action> actions = state.action_space();
        if (state.turn == TicTacToe::PLAYER_X)
        {
            // TODO //////////////////////////////////////////////////////////////////////////
            // Implement value estimation for TicTacToe states
            static std::unordered_map<int, float> state_values;

            // Initialize state values if not done already
            if (state_values.empty()) {
                for (int i = 0; i < (1 << 18); ++i) {
                    state_values[i] = 0.5;  // Initialize with 0.5 (neutral value)
                }
            }

            // Choose the action that leads to the state with the highest value
            vector<float> action_values;

            for (const auto& action : actions) {
                TicTacToe::State next_state = state;
                next_state.put(action);
                
                float state_value;
                if (next_state.test_win()) {
                    state_value = 1.0;  // Winning state has the highest value
                } else if (next_state.full()) {
                    state_value = 0.5;  // Draw state has neutral value
                } else {
                    vector<TicTacToe::Action> opponent_actions = next_state.action_space();
                    TicTacToe::Action opponent_action = opponent_actions[0];
                    next_state.put(opponent_action);
                    state_value = state_values[next_state.board];
                }

                action_values.push_back(state_value);
            }

            // Calculate probabilities using softmax
            float temperature = 0.1;  // Adjust this value to control exploration/exploitation
            vector<float> probabilities(action_values.size());
            float sum_exp = 0.0;
            for (size_t i = 0; i < action_values.size(); ++i) {
                probabilities[i] = exp(action_values[i] / temperature);
                sum_exp += probabilities[i];
            }
            for (float& prob : probabilities) {
                prob /= sum_exp;
            }

            // Choose action based on probabilities
            float random_value = static_cast<float>(rand()) / RAND_MAX;
            float cumulative_prob = 0.0;
            size_t chosen_action_index = 0;
            for (size_t i = 0; i < probabilities.size(); ++i) {
                cumulative_prob += probabilities[i];
                if (random_value <= cumulative_prob) {
                    chosen_action_index = i;
                    break;
                }
            }

            TicTacToe::Action chosen_action = actions[chosen_action_index];

            // Update the value of the current state
            float chosen_value = action_values[chosen_action_index];
            float gamma = 0.8; // Discount factor, the faster the game ends, the more important the future value is
            float alpha = 0.1; // Learning rate
            float future_value = chosen_value * gamma;
            state_values[state.board] = state_values[state.board] + alpha * (future_value - state_values[state.board]);
            if (verbose)
            {
                printf("Probabilities: ");
                for (auto &p : probabilities) 
                    printf("%f, ", p);
                printf("\n");
            }

            return chosen_action;
            // TODO //////////////////////////////////////////////////////////////////////////
        }
        else
        {
            return actions[0];
        }
    }
    TicTacToePolicyDefault() {}
};

int main()
{
    srand(time(nullptr)); // Initialize random seed

    verbose = false;

    // Training Phase
    {
        TicTacToePolicyDefault training_policy;
        for (int i = 0; i < training_iterations; ++i)
        {
            TicTacToe env(false); // Quiet mode during training
            bool done = false;

            while (!done)
            {
                TicTacToe::State state = env.get_state();
                TicTacToe::Action action = training_policy(state);
                env.step(action);
                done = env.done();
            }
        }

        cout << "Training completed after " << training_iterations << " iterations." << endl;
    }

    // Evaluation Phase
    verbose = true;

    {
        TicTacToePolicyDefault policy;
        TicTacToe env(true); // Verbose mode for evaluation
        bool done = false;
        while (!done)
        {
            TicTacToe::State state = env.get_state();
            TicTacToe::Action action = policy(state);
            env.step(action);
            done = env.done();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        int winner = env.winner();
        if (winner == TicTacToe::PLAYER_X)
            cout << "Player X wins!" << endl;
        else if (winner == TicTacToe::PLAYER_O)
            cout << "Player O wins!" << endl;
        else
            cout << "It's a draw!" << endl;
    }

    return 0;
}