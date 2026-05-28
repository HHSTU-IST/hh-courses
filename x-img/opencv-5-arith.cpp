#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 创建两个测试图像
    cv::Mat img1 = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::Mat img2 = cv::Mat::zeros(400, 400, CV_8UC3);

    // 图像1：左红右蓝
    img1(cv::Rect(0, 0, 200, 400)).setTo(cv::Scalar(0, 0, 255));   // 红
    img1(cv::Rect(200, 0, 200, 400)).setTo(cv::Scalar(255, 0, 0)); // 蓝

    // 图像2：上绿下黄
    img2(cv::Rect(0, 0, 400, 200)).setTo(cv::Scalar(0, 255, 0));     // 绿
    img2(cv::Rect(0, 200, 400, 200)).setTo(cv::Scalar(0, 255, 255)); // 黄

    // ===== 加法运算 =====

    // 1. 普通加法（可能溢出，模256）
    cv::Mat add_simple = img1 + img2;

    // 2. cv::add 饱和加法（超过255置为255）
    cv::Mat add_sat;
    cv::add(img1, img2, add_sat);

    // 3. 带权重的加法（融合/水印效果）
    cv::Mat add_weighted;
    double alpha = 0.7, beta = 0.3, gamma = 0;
    cv::addWeighted(img1, alpha, img2, beta, gamma, add_weighted);

    // 4. 加上标量（亮度调整）
    cv::Mat add_scalar;
    cv::add(img1, cv::Scalar(50, 50, 50), add_scalar);

    // ===== 乘法运算 =====

    // 创建渐变图像用于乘法演示
    cv::Mat gradient(400, 400, CV_8UC1);
    for (int i = 0; i < 400; i++)
    {
        for (int j = 0; j < 400; j++)
        {
            gradient.at<uchar>(i, j) = static_cast<uchar>((i / 400.0) * 255);
        }
    }
    cv::Mat gradient_3c;
    cv::cvtColor(gradient, gradient_3c, cv::COLOR_GRAY2BGR);

    // 1. 标量乘法（对比度调整）
    cv::Mat mul_scalar;
    cv::multiply(img1, cv::Scalar(1.5, 1.5, 1.5), mul_scalar, 1, CV_8UC3);
    // 或使用 convertTo
    // img1.convertTo(mul_scalar, CV_8UC3, 1.5, 0);

    // 2. 图像乘法（掩膜效果）
    cv::Mat mul_image;
    cv::multiply(img1, gradient_3c, mul_image, 1.0 / 255.0);

    // 3. 点乘（元素级乘法，需要浮点类型）
    cv::Mat img1_float, grad_float, mul_float;
    img1.convertTo(img1_float, CV_32F, 1.0 / 255.0);
    gradient_3c.convertTo(grad_float, CV_32F, 1.0 / 255.0);
    cv::multiply(img1_float, grad_float, mul_float);
    mul_float.convertTo(mul_image, CV_8U, 255);

    // 显示结果
    cv::imshow("Image 1", img1);
    cv::imshow("Image 2", img2);
    cv::imshow("Simple Add (mod 256)", add_simple);
    cv::imshow("Saturated Add", add_sat);
    cv::imshow("Weighted Add", add_weighted);
    cv::imshow("Add Scalar (Brightness)", add_scalar);
    cv::imshow("Multiply Scalar (Contrast)", mul_scalar);
    cv::imshow("Multiply with Gradient", mul_image);

    // 实际应用：图像融合
    cv::Mat lena = cv::imread("lena.jpg");
    cv::Mat logo = cv::imread("opencv_logo.png");
    if (!lena.empty() && !logo.empty())
    {
        cv::resize(logo, logo, cv::Size(lena.cols / 4, lena.rows / 4));
        cv::Mat roi = lena(cv::Rect(lena.cols - logo.cols - 10,
                                    10, logo.cols, logo.rows));
        cv::addWeighted(roi, 0.8, logo, 0.2, 0, roi);
        cv::imshow("Watermarked", lena);
    }

    cv::waitKey(0);
    return 0;
}
