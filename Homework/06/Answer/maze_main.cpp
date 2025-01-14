#include <ctime>
#include <fstream>
#include "maze.hpp"

class MazePolicyBase
{
public:
    virtual int operator()(const MazeEnv::State &state) const = 0;
};

class MazePolicyQLearning : public MazePolicyBase
{
public:
    int operator()(const MazeEnv::State &state) const
    {
        int best_action = 0;
        double best_value = q[locate(state, 0)];
        double q_s_a;
        for (int action = 1; action < 4; ++action)
        {
            q_s_a = q[locate(state, action)];
            if (q_s_a > best_value)
            {
                best_value = q_s_a;
                best_action = action;
            }
        }
        return best_action;
    }

    MazePolicyQLearning(const MazeEnv &e) : env(e)
    {
        epsilon = 0.1;
        alpha = 0.1;
        gamma = 0.95;
        q = new double[e.max_x * e.max_y * 4];
        srand(2022);
        for (int i = 0; i < e.max_x * e.max_y * 4; ++i)
        {
            q[i] = 1.0 / (rand() % (e.max_x * e.max_y) + 1);
        }
    }

    ~MazePolicyQLearning()
    {
        delete[] q;
    }

    void learn(int iter = 10000, int verbose_freq = 1)
    {
        bool done;
        int action, next_action;
        double reward;
        int episode_step;
        MazeEnv::State state, next_state;
        MazeEnv::StepResult step_result;

        for (int i = 0; i < iter; ++i)
        {
            state = env.reset();
            done = false;
            episode_step = 0;
            while (not done)
            {
                action = epsilon_greedy(state);
                step_result = env.step(action);
                next_state = step_result.next_state;
                reward = step_result.reward;
                done = step_result.done;
                ++episode_step;
                next_action = (*this)(next_state);
                q[locate(state, action)] += alpha * (gamma * q[locate(next_state, next_action)] + reward - q[locate(state, action)]);
                state = next_state;
            }
            if (i % verbose_freq == 0)
            {
                cout << "episode_step: " << episode_step << endl;
            }
        }
    }

    int epsilon_greedy(MazeEnv::State state) const
    {
        if (rand() % 100000 < epsilon * 100000)
        {
            return rand() % 4;
        }
        return (*this)(state);
    }

    inline int locate(MazeEnv::State state, int action) const
    {
        return state.second * env.max_x * 4 + state.first * 4 + action;
    }

    void print_policy() const
    {
        static const char action_vis[] = "<>v^";
        int action;
        MazeEnv::State state;
        for (int i = 0; i < env.max_y; ++i)
        {
            for (int j = 0; j < env.max_x; ++j)
            {
                state = MazeEnv::State(j, i);
                if (not env.is_valid_state(state))
                {
                    cout << "#";
                }
                else if (env.is_goal_state(state))
                {
                    cout << "G";
                }
                // Add this condition to check for start state
                else if (j == env.start_x && i == env.start_y)
                {
                    cout << "S";
                }
                else
                {
                    action = (*this)(MazeEnv::State(j, i));
                    cout << action_vis[action];
                }
            }
            cout << endl;
        }
        cout << endl;
    }

private:
    MazeEnv env;
    double *q;
    double epsilon, alpha, gamma;
};

class MazePolicyDynaQ : public MazePolicyBase
{
public:
    MazePolicyDynaQ(const MazeEnv &e, bool use_dyna_plus = false) : env(e), another_env(e), is_dyna_plus(use_dyna_plus)
    {
        epsilon = 0.1;
        alpha = 0.1;
        gamma = 0.9;
        kappa = 0.1;         // Dyna-Q+ bonus parameter
        planning_steps = 50; // Number of planning steps per real step

        q = new double[e.max_x * e.max_y * 4];
        model = new ModelEntry[e.max_x * e.max_y * 4];

        srand(2022);
        for (int i = 0; i < e.max_x * e.max_y * 4; ++i)
        {
            q[i] = 1.0 / (rand() % (e.max_x * e.max_y) + 1);
        }
    }

    MazePolicyDynaQ(const MazeEnv &e, const MazeEnv &another_env, bool use_dyna_plus = false,
                    int switch_episode = 1000, string exp_type = "Default")
        : env(e), another_env(another_env), is_dyna_plus(use_dyna_plus),
          experiment_type(exp_type), switch_episode(switch_episode)
    {
        is_switch = true;

        epsilon = 0.1;
        alpha = 0.0009;
        gamma = 0.8;
        kappa = 0.0001;      // Dyna-Q+ bonus parameter
        planning_steps = 50; // Number of planning steps per real step

        q = new double[e.max_x * e.max_y * 4];
        model = new ModelEntry[e.max_x * e.max_y * 4];

        srand(2022);
        for (int i = 0; i < e.max_x * e.max_y * 4; ++i)
        {
            q[i] = 1.0 / (rand() % (e.max_x * e.max_y) + 1);
        }
    }

    ~MazePolicyDynaQ()
    {
        delete[] q;
        delete[] model;
    }

    int operator()(const MazeEnv::State &state) const override
    {
        int best_action = 0;
        double best_value = q[locate(state, 0)];
        for (int action = 1; action < 4; ++action)
        {
            double q_s_a = q[locate(state, action)];
            if (q_s_a > best_value)
            {
                best_value = q_s_a;
                best_action = action;
            }
        }
        return best_action;
    }

    void learn(int iter = 10000, int verbose_freq = 1)
    {
        // Create output file name based on policy type and experiment
        string filename = (is_dyna_plus ? "MazePolicyDynaQ_plus-" : "MazePolicyDynaQ-") +
                          experiment_type + ".txt";
        ofstream outFile(filename);

        int time_step = 0;
        double total_accumulated_reward = 0.0;
        for (int i = 0; i < iter; ++i)
        {
            if (is_switch && i == switch_episode)
            {
                switch_env(another_env);
            }

            MazeEnv::State state = env.reset();
            bool done = false;
            int episode_step = 0;
            vector<double> one_episode_reward; // Track episode rewards

            while (not done)
            {
                time_step++;
                // Direct RL
                int action = epsilon_greedy(state);
                int state_action_idx = locate(state, action);

                MazeEnv::StepResult result = env.step(action);
                MazeEnv::State next_state = result.next_state;
                double reward;
                if (is_dyna_plus)
                    reward = result.reward + kappa * sqrt(time_step - model[state_action_idx].last_visited_time);
                else
                    reward = result.reward;
                one_episode_reward.push_back(reward); // Accumulate rewards
                done = result.done;

                // Update Q-value
                int next_action = (*this)(next_state);
                q[state_action_idx] += alpha * (reward + gamma * q[locate(next_state, next_action)] - q[state_action_idx]);

                // Update model
                model[state_action_idx] = {next_state, reward, true, time_step};

                // Planning
                for (int p = 0; p < planning_steps; ++p)
                {
                    // Random previously observed state and action
                    MazeEnv::State plan_state(rand() % env.max_x, rand() % env.max_y);
                    int plan_action = rand() % 4;
                    int plan_idx = locate(plan_state, plan_action);

                    ModelEntry &entry = model[plan_idx];
                    if (entry.visited) // Only plan if we've seen this state-action
                    {
                        double plan_reward = entry.reward;
                        if (is_dyna_plus)
                        {
                            // Add exploration bonus for Dyna-Q+
                            plan_reward += kappa * sqrt(time_step - entry.last_visited_time);
                        }

                        int next_best_action = (*this)(entry.next_state);
                        q[plan_idx] += alpha * (plan_reward +
                                                gamma * q[locate(entry.next_state, next_best_action)] -
                                                q[plan_idx]);
                    }
                }

                state = next_state;
                episode_step++;
            }

            // Save accumulated reward for this episode
            double one_episode_accumulated_reward = 0.0;
            for (auto it = one_episode_reward.rbegin(); it != one_episode_reward.rend(); ++it)
            {
                one_episode_accumulated_reward = *it + one_episode_accumulated_reward * gamma;
            }
            total_accumulated_reward += one_episode_accumulated_reward;
            outFile << i << " " << total_accumulated_reward << endl;
        }

        outFile.close();
    }

    void print_policy() const
    {
        static const char action_vis[] = "<>v^";
        int action;
        MazeEnv::State state;
        for (int i = 0; i < env.max_y; ++i)
        {
            for (int j = 0; j < env.max_x; ++j)
            {
                state = MazeEnv::State(j, i);
                if (not env.is_valid_state(state))
                {
                    cout << "#";
                }
                else if (env.is_goal_state(state))
                {
                    cout << "G";
                }
                // Add this condition to check for start state
                else if (j == env.start_x && i == env.start_y)
                {
                    cout << "S";
                }
                else
                {
                    action = (*this)(MazeEnv::State(j, i));
                    cout << action_vis[action];
                }
            }
            cout << endl;
        }
        cout << endl;
    }

    void switch_env(const MazeEnv &e)
    {
        env = e;
    }

private:
    struct ModelEntry
    {
        MazeEnv::State next_state;
        double reward;
        bool visited = false;
        int last_visited_time = 0;
    };

    bool is_switch;     // 是否切换环境
    int switch_episode; // 切换环境的时间步, learn 到第 switch_episode 步时切换环境

    MazeEnv env, another_env;
    double *q;
    ModelEntry *model;
    double epsilon, alpha, gamma, kappa;
    int planning_steps;
    bool is_dyna_plus;
    string experiment_type;

    int epsilon_greedy(MazeEnv::State state) const
    {
        if (rand() % 100000 < epsilon * 100000)
        {
            return rand() % 4;
        }
        return (*this)(state);
    }

    inline int locate(MazeEnv::State state, int action) const
    {
        return state.second * env.max_x * 4 + state.first * 4 + action;
    }
};

int main()
{
    // // 测试环境
    // const int max_x = 9, max_y = 6;
    // const int start_x = 0, start_y = 2;
    // const int target_x = 8, target_y = 0;
    // int maze[max_y][max_x] = {
    //     {0, 0, 0, 0, 0, 0, 0, 1, 0},
    //     {0, 0, 1, 0, 0, 0, 0, 1, 0},
    //     {0, 0, 1, 0, 0, 0, 0, 1, 0},
    //     {0, 0, 1, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 1, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0}};
    // MazeEnv env(maze, max_x, max_y, start_x, start_y, target_x, target_y);

    // env.reset();
    // MazePolicyQLearning policy(env);
    // policy.learn(10000, 10000);
    // policy.print_policy();

    // env.reset();
    // MazePolicyDynaQ policy_dyna(env);
    // policy_dyna.learn(10000, 10000);
    // policy_dyna.print_policy();

    // env.reset();
    // MazePolicyDynaQ policy_dyna_plus(env, true);
    // policy_dyna_plus.learn(10000, 10000);
    // policy_dyna_plus.print_policy();

    // 实验环境
    const int max_x_test = 9, max_y_test = 6;
    const int start_x_test = 3, start_y_test = 0;
    const int target_x_test = 8, target_y_test = 5;
    int maze_1[max_y_test][max_x_test] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0}};
    int maze_2[max_y_test][max_x_test] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0}};
    int maze_3[max_y_test][max_x_test] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0}};
    MazeEnv env_1(maze_1, max_x_test, max_y_test, start_x_test, start_y_test, target_x_test, target_y_test);
    MazeEnv env_2(maze_2, max_x_test, max_y_test, start_x_test, start_y_test, target_x_test, target_y_test);
    MazeEnv env_3(maze_3, max_x_test, max_y_test, start_x_test, start_y_test, target_x_test, target_y_test);

    // Print Maze
    env_1.print_maze();
    env_2.print_maze();
    env_3.print_maze();

    // 实验 Blocking Maze
    MazePolicyDynaQ policy_blocking(env_1, env_2, false, 1000, "Blocking_Maze");
    policy_blocking.learn(3000, 3000);

    MazePolicyDynaQ policy_blocking_plus(env_1, env_2, true, 1000, "Blocking_Maze");
    policy_blocking_plus.learn(3000, 3000);

    // 实验 Shortcut Maze
    MazePolicyDynaQ policy_shortcut(env_2, env_3, false, 3000, "Shortcut_Maze");
    policy_shortcut.learn(6000, 6000);

    MazePolicyDynaQ policy_shortcut_plus(env_2, env_3, true, 3000, "Shortcut_Maze");
    policy_shortcut_plus.learn(6000, 6000);

    return 0;
}
