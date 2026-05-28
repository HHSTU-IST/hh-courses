#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 创建 800x600 的白色画布 (BGR: 255,255,255)
    cv::Mat canvas = cv::Mat::zeros(600, 800, CV_8UC3);
    canvas.setTo(cv::Scalar(255, 255, 255));

    // 1. 绘制圆形 (圆心, 半径, 颜色, 线宽, 线型)
    // 圆心 (400, 150), 半径 80, 红色, 线宽 3
    cv::circle(canvas, cv::Point(400, 150), 80,
               cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

    // 填充圆形 (线宽为负数或 CV_FILLED)
    cv::circle(canvas, cv::Point(200, 150), 50,
               cv::Scalar(255, 0, 0), -1);

    // 2. 绘制矩形 (左上角, 右下角, 颜色, 线宽)
    // 左上角 (550, 100), 右下角 (700, 250), 绿色
    cv::rectangle(canvas, cv::Point(550, 100), cv::Point(700, 250),
                  cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    // 填充矩形
    cv::rectangle(canvas, cv::Point(50, 100), cv::Point(150, 200),
                  cv::Scalar(255, 255, 0), -1);

    // 3. 绘制多边形（三角形）
    std::vector<cv::Point> triangle;
    triangle.push_back(cv::Point(400, 400)); // 顶点A
    triangle.push_back(cv::Point(300, 550)); // 顶点B
    triangle.push_back(cv::Point(500, 550)); // 顶点C

    // 转换为点集数组（polylines需要）
    std::vector<std::vector<cv::Point>> polygons;
    polygons.push_back(triangle);

    // 绘制多边形轮廓 (画布, 点集, 是否闭合, 颜色, 线宽)
    cv::polylines(canvas, polygons, true,
                  cv::Scalar(255, 0, 255), 3, cv::LINE_AA);

    // 绘制填充多边形（四边形）
    std::vector<cv::Point> quadrilateral;
    quadrilateral.push_back(cv::Point(600, 350));
    quadrilateral.push_back(cv::Point(750, 380));
    quadrilateral.push_back(cv::Point(720, 500));
    quadrilateral.push_back(cv::Point(580, 480));

    std::vector<std::vector<cv::Point>> quad_poly;
    quad_poly.push_back(quadrilateral);

    // 填充多边形
    cv::fillPoly(canvas, quad_poly, cv::Scalar(128, 0, 128));

    // 显示结果
    cv::imshow("Shapes Demo", canvas);
    cv::waitKey(0);
    // cv::imwrite("output_shapes.png", canvas);

    return 0;
}
