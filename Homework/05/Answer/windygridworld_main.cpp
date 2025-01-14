#include <utility>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;
class WindyGridWorld
{
public:
    static const int
        LEFT = 0,
        RIGHT = 1, UP = 2, DOWN = 3;
    static const char ACTION_NAME[][16];
    static const int WIND_X[];
    typedef pair<int, int> State;
    bool verbose;
    State state()
    {
        return make_pair(x, y);
    }
    void set_state(int x, int y)
    {
        this->x = x;
        this->y = y;
        if (verbose)
        {
            cout << verbose << endl;
            cout << "State reset: (" << x << "," << y << ")" << endl;
        }
    }
    void reset()
    {
        set_state(0, 3);
    }
    pair<State, double> step(int action)
    {
        State old_state = state();
        double reward = state_transition(action);
        if (verbose)
        {
            cout << "State: (" << old_state.first << "," << old_state.second << ")" << endl;
            cout << "Action: " << ACTION_NAME[action] << endl;
            cout << "Wind: (0," << WIND_X[old_state.first] << ")" << endl;
            cout << "Reward: " << reward << endl;
            cout << "New State: (" << x << "," << y << ")" << endl
                 << endl;
        }
        return make_pair(state(), reward);
    }
    int sample_action() const
    {
        return rand() % 4;
    }
    bool done() const
    {
        return x == 7 and y == 3;
    }
    WindyGridWorld(int x = 0, int y = 3, bool verbose = false)
    {
        this->verbose = verbose;
        set_state(x, y);
    }

private:
    int x, y;

    double state_transition(int action)
    {
        int new_x = x, new_y = y;
        switch (action)
        {
        case LEFT:
            --new_x;
            break;
        case RIGHT:
            ++new_x;
            break;
        case UP:
            ++new_y;
            break;
        case DOWN:
            --new_y;
            break;
        }
        new_y += WIND_X[x];
        x = max(0, new_x);
        x = min(x, 10 - 1);
        y = max(0, new_y);
        y = min(y, 7 - 1);
        return -1;
    }
};
const char WindyGridWorld::ACTION_NAME[][16] = {"LEFT(-1,0)", "RIGHT(1,0)", "UP(0,1)", "DOWN(0,-1)"};
const int WindyGridWorld::WIND_X[] = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};

class WindyGridWorldPolicyBase
{
public:
    virtual int operator()(const WindyGridWorld::State &state) const = 0;
    void print_path(void) const
    {
        WindyGridWorld env = WindyGridWorld();
        WindyGridWorld::State state;
        int episode_len = 0;
        while (not env.done())
        {
            state = env.state();
            cout << "(" << state.first << "," << state.second << ")->";
            env.step((*this)(state));
            ++episode_len;
        }
        cout << "(" << env.state().first << "," << env.state().second << ")." << endl;
        cout << "Episode length: " << episode_len << endl;
    }
};

class WindyGridWorldPolicySarsa : public WindyGridWorldPolicyBase
{
public:
    WindyGridWorldPolicySarsa()
    {
        epsilon = 0.1;
        alpha = 0.5;
        gamma = 1.0;
        memset(q, 0, sizeof(q));
    }
    virtual int operator()(const WindyGridWorld::State &state) const
    {
        int best_action = 0;
        int x = state.first, y = state.second;
        double best_value = q[y][x][0];
        for (int i = 1; i < 4; ++i)
        {
            if (q[y][x][i] > best_value)
            {
                best_value = q[y][x][i];
                best_action = i;
            }
        }
        return best_action;
    }
    void learn(int iter = 1000000)
    {
        WindyGridWorld env;
        for (int i = 0; i < iter; ++i)
        {
            env.reset();
            WindyGridWorld::State state = env.state();
            int action = epsilon_greedy(state);

            while (!env.done())
            {
                // Take action, observe reward and next state
                auto [next_state, reward] = env.step(action);
                // Choose next action using epsilon-greedy
                int next_action = epsilon_greedy(next_state);

                // SARSA update
                int x = state.first, y = state.second;
                int next_x = next_state.first, next_y = next_state.second;
                q[y][x][action] += alpha * (reward + gamma * q[next_y][next_x][next_action] - q[y][x][action]);

                state = next_state;
                action = next_action;
            }
        }
    }

    void print_path(void) const
    {
        cout << "Sarsa result:" << endl;
        this->WindyGridWorldPolicyBase::print_path();
    }

private:
    double q[7][10][4];
    double epsilon, alpha, gamma;
    int epsilon_greedy(const WindyGridWorld::State &state)
    {
        if (rand() % 100000 < epsilon * 100000)
        {
            return rand() % 4;
        }
        return (*this)(state);
    }
};

class WindyGridWorldPolicyQLearning : public WindyGridWorldPolicyBase
{
public:
    WindyGridWorldPolicyQLearning()
    {
        epsilon = 0.1;
        alpha = 0.5;
        gamma = 1.0;
        memset(q, 0, sizeof(q));
    }
    virtual int operator()(const WindyGridWorld::State &state) const
    {
        int best_action = 0;
        int x = state.first, y = state.second;
        double best_value = q[y][x][0];
        for (int i = 1; i < 4; ++i)
        {
            if (q[y][x][i] > best_value)
            {
                best_value = q[y][x][i];
                best_action = i;
            }
        }
        return best_action;
    }

    void learn(int iter = 1000000)
    {
        WindyGridWorld env;
        for (int i = 0; i < iter; ++i)
        {
            env.reset();
            WindyGridWorld::State state = env.state();

            while (!env.done())
            {
                // Choose action using epsilon-greedy
                int action = epsilon_greedy(state);
                // Take action, observe reward and next state
                auto [next_state, reward] = env.step(action);

                // Q-Learning update (using max Q-value of next state)
                int x = state.first, y = state.second;
                int next_x = next_state.first, next_y = next_state.second;

                // Find max Q-value for next state
                double max_next_q = q[next_y][next_x][0];
                for (int a = 1; a < 4; ++a)
                {
                    max_next_q = max(max_next_q, q[next_y][next_x][a]);
                }

                q[y][x][action] += alpha * (reward + gamma * max_next_q - q[y][x][action]);
                state = next_state;
            }
        }
    }

    void print_path(void) const
    {
        cout << "Q learning result:" << endl;
        this->WindyGridWorldPolicyBase::print_path();
    }

private:
    double q[7][10][4];
    double epsilon, alpha, gamma;
    int epsilon_greedy(const WindyGridWorld::State &state)
    {
        if (rand() % 100000 < epsilon * 100000)
        {
            return rand() % 4;
        }
        return (*this)(state);
    }
};

#include <chrono>
#include <thread>

int main()
{
    WindyGridWorldPolicySarsa policy_sarsa;
    policy_sarsa.learn();
    policy_sarsa.print_path();

    WindyGridWorldPolicyQLearning policy_q;
    policy_q.learn();
    policy_q.print_path();
    return 0;
}
