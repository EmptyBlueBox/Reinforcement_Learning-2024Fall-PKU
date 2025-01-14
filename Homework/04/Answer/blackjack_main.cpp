#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <utility>
#include <vector>
#include "blackjack.hpp"
#include <random>

class BlackjackPolicyBase
{
public:
    virtual int operator()(const Blackjack::State &state) = 0;
};

class BlackjackPolicyDefault : public BlackjackPolicyBase
{
public:
    int operator()(const Blackjack::State &state)
    {
        if (state.turn == Blackjack::PLAYER)
        {
            return state.player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
        }
        else
        {
            return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
        }
    }
};

class BlackjackPolicyLearnableDefault : public BlackjackPolicyBase
{
    static constexpr const char *ACTION_NAME = "HS";

public:
    int operator()(const Blackjack::State &state)
    {
        if (state.turn == Blackjack::DEALER)
        {
            return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
        }
        else
        {
            return policy[state.dealer_shown][state.player_ace][state.player_sum];
        }
    }
    void update_policy()
    {
        for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
        {
            for (int player_ace = 0; player_ace <= 1; ++player_ace)
            {
                for (int player_sum = 11; player_sum <= 21; ++player_sum)
                {
                    int best_action = 0;
                    double best_value = value[dealer_shown][player_ace][player_sum][0];
                    for (int action = 0; action <= 1; ++action)
                    {
                        if (value[dealer_shown][player_ace][player_sum][action] > best_value)
                        {
                            best_value = value[dealer_shown][player_ace][player_sum][action];
                            best_action = action;
                        }
                    }
                    policy[dealer_shown][player_ace][player_sum] = best_action;
                }
            }
        }
    }

    // REMEMBER to call set_value_initial() at the beginning
    // simulate from every possible initial state:(dealer_shown, player_ace, player_sum) \
        (call Blackjack::reset(,,) to do this) and player's every possible first action
    // BE AWARE only use player's steps (rather than dealer's) to update value estimation.
    void update_value(Blackjack &env, int n = 10000)
    {
        set_value_initial();

        // Replace random initialization with nested loops
        for (int iter = 0; iter < n; ++iter)
        {
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                for (int player_ace = 0; player_ace <= 1; ++player_ace)
                {
                    for (int player_sum = 11; player_sum <= 21; ++player_sum)
                    {
                        for (int first_action = 0; first_action <= 1; ++first_action)
                        {
                            env.reset(dealer_shown, player_ace, player_sum);
                            episode.clear();
                            bool done = false;
                            bool is_first_action = true;

                            while (!done)
                            {
                                Blackjack::State state = env.state();

                                int player_action;
                                if (is_first_action)
                                {
                                    player_action = first_action; // Use loop variable for the first action
                                    is_first_action = false;
                                }
                                else
                                {
                                    player_action = policy[state.dealer_shown][state.player_ace][state.player_sum]; // Use policy for subsequent actions
                                }

                                int dealer_action = (state.dealer_sum >= 17) ? Blackjack::STICK : Blackjack::HIT;
                                int current_action = (state.turn == Blackjack::PLAYER) ? player_action : dealer_action;
                                Blackjack::StepResult result = env.step(current_action);

                                if (state.turn == Blackjack::PLAYER)
                                {
                                    episode.emplace_back(state, player_action);
                                }

                                done = result.done;
                            }

                            // Determine the reward based on the final state
                            double reward = 0.0;
                            const Blackjack::State &final_state = env.state();

                            if (final_state.player_sum > 21)
                            {
                                reward = -1.0; // Player busts
                            }
                            else if (final_state.dealer_sum > 21)
                            {
                                reward = 1.0; // Dealer busts
                            }
                            else if (final_state.player_sum > final_state.dealer_sum)
                            {
                                reward = 1.0; // Player wins
                            }
                            else if (final_state.player_sum < final_state.dealer_sum)
                            {
                                reward = -1.0; // Dealer wins
                            }
                            // reward remains 0.0 for a draw

                            // Update value estimates based on the episode
                            for (const auto &step : episode)
                            {
                                int &count = state_action_count[step.dealer_shown][step.player_ace][step.player_sum][step.action];
                                double &val = value[step.dealer_shown][step.player_ace][step.player_sum][step.action];
                                count++;
                                val += (reward - val) / count; // Incremental mean update
                            }

                            update_policy();
                        }
                    }
                }
            }
        }
    }

    void print_policy() const
    {
        cout << setw(10) << "Player Without Ace" << "\t\t" << "Player With Usable Ace." << endl;
        for (int player_sum = 21; player_sum >= 11; --player_sum)
        {
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                cout << ACTION_NAME[policy[dealer_shown][0][player_sum]];
            }
            cout << "\t\t\t\t";
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                cout << ACTION_NAME[policy[dealer_shown][1][player_sum]];
            }
            cout << endl;
        }
        cout << endl;
    }
    void print_value() const
    {
        cout << setw(40) << "Player Without Ace" << setw(20) << "\t\t\t" << "Player With Usable Ace." << endl;
        for (int player_sum = 21; player_sum >= 11; --player_sum)
        {
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][0][player_sum][policy[dealer_shown][0][player_sum]];
            }
            cout << "\t";
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][1][player_sum][policy[dealer_shown][1][player_sum]];
            }
            cout << endl;
        }
        cout << endl;
    }

    BlackjackPolicyLearnableDefault()
    {
        for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
        {
            for (int player_ace = 0; player_ace <= 1; ++player_ace)
            {
                for (int player_sum = 0; player_sum <= 21; ++player_sum)
                {
                    int &action = policy[dealer_shown][player_ace][player_sum];
                    action = player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
                }
            }
        }
    }

private:
    // 11: dealer_shown (A-10);
    // 2: player_usable_ace (true/false);
    // 22: player_sum (only need to consider 11-20, because HIT when sum<11 and STICK when sum=21 are always best action.)
    int policy[11][2][22];
    // 11:dealer_shown; 2:player_usable_ace; 22:player_sum; 2:action (0:HIT, 1:STICK)
    double value[11][2][22][2];
    int state_action_count[11][2][22][2];

    // record a episode sampled (only player's steps).
    struct EpisodeStep
    {
        // state: (dealer_shown, player_ace, player_sum)
        int dealer_shown;
        int player_ace;
        int player_sum;
        // the action taken at state
        int action;
        EpisodeStep(const Blackjack::State &state, int action)
        {
            dealer_shown = state.dealer_shown;
            player_ace = int(state.player_ace);
            player_sum = state.player_sum;
            this->action = action;
        }
        EpisodeStep() {}
    };
    vector<EpisodeStep> episode;

    void set_value_initial()
    {
        memset(value, 0, sizeof(value));
        memset(state_action_count, 0, sizeof(state_action_count));
    }
};

// Demonstrative play of player & dealer with default policy
int main()
{
    Blackjack env(false);
    BlackjackPolicyLearnableDefault policy;
    policy.update_value(env);
    policy.update_policy();
    policy.print_policy();
    return 0;
}
