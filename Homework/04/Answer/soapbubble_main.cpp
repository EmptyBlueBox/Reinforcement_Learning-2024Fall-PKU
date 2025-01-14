#include <iostream>
#include <vector>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace std;

class SoapBubble
{
public:
    typedef vector<vector<double>> Bubble;
    SoapBubble(int max_x, int max_y,
               const vector<int> &x_lower, const vector<int> &x_upper,
               const vector<int> &y_lower, const vector<int> &y_upper,
               const Bubble &bubble)
    {
        SoapBubble(max_x, max_y, x_lower, x_upper, y_lower, y_upper);
        set_bubble(bubble);
    }
    SoapBubble(int max_x, int max_y,
               const vector<int> &x_lower, const vector<int> &x_upper,
               const vector<int> &y_lower, const vector<int> &y_upper)
    {
        this->max_x = max_x;
        this->max_y = max_y;
        this->x_lower = x_lower;
        this->x_upper = x_upper;
        this->y_lower = y_lower;
        this->y_upper = y_upper;
    }
    void set_bubble(const Bubble &bubble)
    {
        this->bubble = bubble;
        for (int y = 0; y < max_y; ++y)
        {
            for (int x = 0; x < max_x; ++x)
            {
                if (not at_border(x, y))
                {
                    this->bubble[y][x] = 0;
                }
            }
        }
    }
    bool inside_bubble(int x, int y) const
    {
        return x_lower[y] <= x and x <= x_upper[y] and y_lower[x] <= y and y <= y_upper[x];
    }
    bool at_border(int x, int y) const
    {
        return inside_bubble(x, y) and (x_lower[y] == x or x == x_upper[y] or y_lower[x] == y or y == y_upper[x]);
    }
    void print_bubble(void) const
    {
        for (int y = 0; y < max_y; ++y)
        {
            for (int x = 0; x < max_x; ++x)
            {
                if (inside_bubble(x, y))
                {
                    cout << setw(8) << setprecision(2) << bubble[y][x];
                }
                else
                {
                    cout << setw(8) << " ";
                }
            }
            cout << endl;
        }
    }

    void inner_heights_dp(int iter = 500000)
    {
        for (int i = 0; i < iter; ++i)
        {
            for (int y = 0; y < max_y; ++y)
            {
                for (int x = 0; x < max_x; ++x)
                {
                    if (inside_bubble(x, y) and not at_border(x, y))
                    {
                        double average = 0;
                        int inside_neighbor = 0;
                        for (int d = 0; d < DIR_NUM; ++d)
                        {
                            int new_x = x + DX[d];
                            int new_y = y + DY[d];
                            if (inside_bubble(new_x, new_y))
                            {
                                ++inside_neighbor;
                                average += bubble[new_y][new_x];
                            }
                        }
                        bubble[y][x] = average / inside_neighbor;
                    }
                }
            }
        }
    }

    void inner_heights_mc(int iter = 500000)
    {
        for (int i = 0; i < iter; ++i)
        {
            int x = rand() % max_x;
            int y = rand() % max_y;

            if (inside_bubble(x, y) && !at_border(x, y))
            {
                double average = 0;
                int inside_neighbor = 0;
                for (int d = 0; d < DIR_NUM; ++d)
                {
                    int new_x = x + DX[d];
                    int new_y = y + DY[d];
                    if (inside_bubble(new_x, new_y))
                    {
                        ++inside_neighbor;
                        average += bubble[new_y][new_x];
                    }
                }
                if (inside_neighbor > 0)
                {
                    bubble[y][x] = average / inside_neighbor;
                }
            }
        }
    }

private:
    static const int DIR_NUM = 4;
    static const int DX[DIR_NUM];
    static const int DY[DIR_NUM];
    int max_x, max_y;
    vector<int> x_lower, x_upper, y_lower, y_upper;
    Bubble bubble;
};
const int SoapBubble::DX[] = {0, 1, 0, -1}, SoapBubble::DY[] = {1, 0, -1, 0};

SoapBubble default_bubble_generator(int max_x = 10, int max_y = 10)
{
    vector<int> x_lower, x_upper, y_lower, y_upper;
    for (int y = 0, lower, upper; y < max_y; ++y)
    {
        lower = y >> 2;
        upper = max_y - 1 - ((max_y - y) >> 2);
        x_lower.push_back(lower);
        x_upper.push_back(upper);
    }
    for (int x = 0, lower, upper; x < max_x; ++x)
    {
        lower = x >> 2;
        upper = max_x - 1 - ((max_x - x) >> 2);
        y_lower.push_back(lower);
        y_upper.push_back(upper);
    }
    SoapBubble soap_bubble(max_x, max_y, x_lower, x_upper, y_lower, y_upper);
    SoapBubble::Bubble bubble(max_y);
    for (int y = 0; y < max_y; ++y)
    {
        for (int x = 0; x < max_x; ++x)
        {
            if (soap_bubble.at_border(x, y))
            {
                bubble[y].push_back(log((sin(x) + 1) / (cos(y) + 1)));
            }
            else
            {
                bubble[y].push_back(0);
            }
        }
    }
    soap_bubble.set_bubble(bubble);
    return soap_bubble;
}

int main()
{
    SoapBubble soap_bubble = default_bubble_generator();
    soap_bubble.print_bubble();
    soap_bubble.inner_heights_dp();
    soap_bubble.print_bubble();
    soap_bubble = default_bubble_generator();
    soap_bubble.inner_heights_mc();
    soap_bubble.print_bubble();
    return 0;
}
