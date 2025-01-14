#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace std;

const int MAX_CARS = 20;
const int MOVE_LIMIT = 5;
const double RENTAL_REWARD = 10.0;
const double MOVE_COST = 2.0;
const double DISCOUNT = 0.9;

const double MEAN_REQUEST_1 = 3.0;
const double MEAN_REQUEST_2 = 4.0;
const double MEAN_RETURN_1 = 3.0;
const double MEAN_RETURN_2 = 2.0;

// Poisson distribution cache
vector<vector<double>> poisson_cache(2 * MAX_CARS + 1, vector<double>(2 * MAX_CARS + 1, -1));

// Poisson distribution with caching
double poisson(int n, double lambda)
{
    if (poisson_cache[n][static_cast<int>(lambda)] < 0)
    {
        poisson_cache[n][static_cast<int>(lambda)] = exp(-lambda) * pow(lambda, n) / tgamma(n + 1);
    }
    return poisson_cache[n][static_cast<int>(lambda)];
}

// Calculate the expected return for a given state and action
double expected_return(const pair<int, int> &state, int action, const vector<vector<double>> &value)
{
    double returns = -MOVE_COST * abs(action);

    int cars1 = min(state.first - action, MAX_CARS);
    int cars2 = min(state.second + action, MAX_CARS);

    // Iterate over all possible rental requests
    for (int rental_request1 = 0; rental_request1 <= 10; rental_request1++)
    {
        for (int rental_request2 = 0; rental_request2 <= 10; rental_request2++)
        {
            double prob = poisson(rental_request1, MEAN_REQUEST_1) * poisson(rental_request2, MEAN_REQUEST_2);

            int valid_rental1 = min(cars1, rental_request1);
            int valid_rental2 = min(cars2, rental_request2);
            double reward = (valid_rental1 + valid_rental2) * RENTAL_REWARD;

            int cars1_after_rent = cars1 - valid_rental1;
            int cars2_after_rent = cars2 - valid_rental2;

            // Iterate over all possible returns
            for (int return1 = 0; return1 <= 10; return1++)
            {
                for (int return2 = 0; return2 <= 10; return2++)
                {
                    double prob_return = poisson(return1, MEAN_RETURN_1) * poisson(return2, MEAN_RETURN_2);
                    int new_cars1 = min(cars1_after_rent + return1, MAX_CARS);
                    int new_cars2 = min(cars2_after_rent + return2, MAX_CARS);
                    double prob_total = prob * prob_return;

                    returns += prob_total * (reward + DISCOUNT * value[new_cars1][new_cars2]);
                }
            }
        }
    }
    return returns;
}

// Perform policy iteration
void policy_iteration()
{
    vector<vector<double>> value(MAX_CARS + 1, vector<double>(MAX_CARS + 1, 0.0));
    vector<vector<int>> policy(MAX_CARS + 1, vector<int>(MAX_CARS + 1, 0));

    bool is_policy_stable = false;
    while (!is_policy_stable)
    {
        // Policy evaluation
        while (true)
        {
            double delta = 0;
            vector<vector<double>> new_value = value;

            // Evaluate the value function for all states
            for (int i = 0; i <= MAX_CARS; i++)
            {
                for (int j = 0; j <= MAX_CARS; j++)
                {
                    double v = value[i][j];
                    new_value[i][j] = expected_return({i, j}, policy[i][j], value);
                    delta = max(delta, abs(v - new_value[i][j]));
                }
            }

            value = new_value;
            if (delta < 1e-4)
            {
                break;
            }
        }

        // Policy improvement
        is_policy_stable = true;
        for (int i = 0; i <= MAX_CARS; i++)
        {
            for (int j = 0; j <= MAX_CARS; j++)
            {
                int old_action = policy[i][j];
                vector<double> action_returns;

                // Evaluate all possible actions
                for (int action = -MOVE_LIMIT; action <= MOVE_LIMIT; action++)
                {
                    if ((0 <= i - action && i - action <= MAX_CARS) && (0 <= j + action && j + action <= MAX_CARS))
                    {
                        action_returns.push_back(expected_return({i, j}, action, value));
                    }
                    else
                    {
                        action_returns.push_back(-INFINITY);
                    }
                }

                // Select the action with the highest expected return
                int new_action = distance(action_returns.begin(), max_element(action_returns.begin(), action_returns.end())) - MOVE_LIMIT;
                policy[i][j] = new_action;

                if (old_action != new_action)
                {
                    is_policy_stable = false;
                }
            }
        }
    }

    // Output the converged policy
    cout << "Policy:" << endl;
    for (const auto &row : policy)
    {
        for (int i = 0; i < row.size(); i++)
        {
            if (i == row.size() - 1)
            {
                cout << row[i];
            }
            else
            {
                cout << row[i] << ",";
            }
        }
        cout << endl;
    }
}

int main()
{
    policy_iteration();
}
