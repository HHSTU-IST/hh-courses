#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat image = cv::imread("lena.jpg");
    if (image.empty())
    {
        image = cv::Mat(400, 400, CV_8UC3);
        cv::putText(image, "Test Image", cv::Point(100, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 3);
        cv::circle(image, cv::Point(200, 200), 150, cv::Scalar(0, 0, 255), 3);
    }

    cv::Mat flip_x, flip_y, flip_xy;

    // flipCode = 0: 沿X轴翻转（垂直翻转）
    cv::flip(image, flip_x, 0);

    // flipCode > 0: 沿Y轴翻转（水平翻转）
    cv::flip(image, flip_y, 1);

    // flipCode < 0: 沿XY轴翻转（180度旋转）
    cv::flip(image, flip_xy, -1);

    // 显示
    cv::imshow("Original", image);
    cv::imshow("Flip X (Vertical)", flip_x);
    cv::imshow("Flip Y (Horizontal)", flip_y);
    cv::imshow("Flip XY (Rotate 180)", flip_xy);

    cv::waitKey(0);
    return 0;
}
