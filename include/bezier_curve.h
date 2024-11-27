#pragma once
#include <vector>

namespace BezierCurve {
struct Point {
    float x;
    float y;
};

void generateCurve(Point start, Point middle, Point end,
                   std::vector<Point>& points);

float getY(std::vector<Point>& points, float x);
}  // namespace BezierCurve
