#include "../include/bezier_curve.h"

float interpolate(float from, float to, float percent) {
    float difference = to - from;
    return from + (difference * percent);
}

void BezierCurve::generateCurve(Point start, Point middle, Point end,
                                std::vector<Point>& points) {
    for (float i = 0; i < 1; i += 0.01) {
        // green line
        float xa = interpolate(start.x, middle.x, i);
        float ya = interpolate(start.y, middle.y, i);
        float xb = interpolate(middle.x, end.x, i);
        float yb = interpolate(middle.y, end.y, i);

        // black dot
        float x = interpolate(xa, xb, i);
        float y = interpolate(ya, yb, i);

        points.push_back({x, y});
    }
}

float linearFunction(BezierCurve::Point p1, BezierCurve::Point p2, float x) {
    float m = (p2.y - p1.y) / (p2.x - p1.x);

    return m * x + p1.y;
}

float BezierCurve::getY(std::vector<Point>& points, float x) {
    for (int i = 0; i < points.size() - 1; i++) {
        if (points[i].x == x) return points[i].y;

        if (points[i].x <= x && points[i + 1].x >= x)
            return linearFunction(points[i], points[i + 1], points[i].x - x);
    }

    return 0.0f;
}