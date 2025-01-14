#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>

const int grid_size = 5;   // 网格大小
const double gamma = 0.9;  // 折扣因子
const double theta = 1e-6; // 停止迭代的阈值

// 动作集合：北、南、西、东
const std::array<std::pair<int, int>, 4> actions = {{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

// 特殊状态及其转换
const std::array<std::pair<std::pair<int, int>, std::pair<int, int>>, 2> special_states = {
    {{{0, 1}, {4, 1}}, {{0, 3}, {2, 3}}}};
const std::array<int, 2> special_rewards = {10, 5};

// 迭代更新价值函数
void policy_evaluation(std::array<std::array<double, grid_size>, grid_size> &v, double gamma, double theta)
{
    while (true)
    {
        double delta = 0;
        std::array<std::array<double, grid_size>, grid_size> new_v = v;
        for (int i = 0; i < grid_size; ++i)
        {
            for (int j = 0; j < grid_size; ++j)
            {
                bool is_special = false;
                for (size_t k = 0; k < special_states.size(); ++k)
                {
                    if (i == special_states[k].first.first && j == special_states[k].first.second)
                    {
                        new_v[i][j] = special_rewards[k] + gamma * v[special_states[k].second.first][special_states[k].second.second];
                        is_special = true;
                        break;
                    }
                }
                if (!is_special)
                {
                    double v_sum = 0;
                    for (const auto &action : actions)
                    {
                        int ni = i + action.first;
                        int nj = j + action.second;
                        if (ni >= 0 && ni < grid_size && nj >= 0 && nj < grid_size)
                        {
                            v_sum += 0.25 * (0 + gamma * v[ni][nj]);
                        }
                        else
                        {
                            v_sum += 0.25 * (-1 + gamma * v[i][j]);
                        }
                    }
                    new_v[i][j] = v_sum;
                }
                delta = std::max(delta, std::abs(new_v[i][j] - v[i][j]));
            }
        }
        v = new_v;
        if (delta < theta)
        {
            break;
        }
    }
}

int main()
{
    std::array<std::array<double, grid_size>, grid_size> v = {0};

    // 计算状态价值
    policy_evaluation(v, gamma, theta);

    // 以矩阵形式输出，保留 3 位小数，并对齐
    for (const auto &row : v)
    {
        for (const auto &val : row)
        {
            std::cout << std::fixed << std::setprecision(3) << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
