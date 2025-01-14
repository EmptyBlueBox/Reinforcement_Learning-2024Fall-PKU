#include <utility>
#include <cstdlib>
#include <iostream>

using namespace std;
class GridWorld
{
public:
    static const int
        NORTH = 0,
        SOUTH = 1, EAST = 2, WEST = 3;
    static const char ACTION_NAME[][16];
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
            cout << "State reset: (" << x << "," << y << ")" << endl;
        }
    }
    void reset()
    {
        set_state(0, 0);
    }
    pair<State, double> step(int action)
    {
        State old_state = state();
        double reward = state_transition(action);
        if (verbose)
        {
            cout << "State: (" << old_state.first << "," << old_state.second << ")" << endl;
            cout << "Action: " << ACTION_NAME[action] << endl;
            cout << "Reward: " << reward << endl;
            cout << "New State: (" << x << "," << y << ")" << endl
                 << endl;
        }
        return make_pair(state(), reward);
    }
    int sample_action()
    {
        return rand() % 4;
    }
    GridWorld(int x = 0, int y = 0, bool verbose = false)
    {
        set_state(x, y);
        this->verbose = verbose;
    }

private:
    int x, y;
    double state_transition(int action)
    {
        if (state() == make_pair(1, 0))
        {
            x = 1;
            y = 4;
            return 10.0;
        }
        if (state() == make_pair(3, 0))
        {
            x = 3;
            y = 2;
            return 5.0;
        }
        if (action == NORTH and y == 0 or
            action == SOUTH and y == 4 or
            action == EAST and x == 4 or
            action == WEST and x == 0)
        {
            return -1.0;
        }
        switch (action)
        {
        case NORTH:
            y--;
            break;
        case SOUTH:
            y++;
            break;
        case EAST:
            x++;
            break;
        case WEST:
            x--;
            break;
        }
        return 0.0;
    }
};
const char GridWorld::ACTION_NAME[][16] = {"NORTH(0,-1)", "SOUTH(0,1)", "EAST:(1,0)", "WEST:(-1,0)"};

#include <chrono>
#include <thread>
int main()
{
    GridWorld env = GridWorld(0, 0, true);
    while (true)
    {
        int action = env.sample_action();
        auto state_reward = env.step(action);
        this_thread::sleep_for(chrono::milliseconds(1000));
    }
    return 0;
}