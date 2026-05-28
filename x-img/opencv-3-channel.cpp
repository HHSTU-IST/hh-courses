#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 读取彩色图像
    cv::Mat image = cv::imread("lena.jpg");
    if (image.empty())
    {
        // 创建测试图像
        image = cv::Mat(400, 400, CV_8UC3, cv::Scalar(100, 150, 200));
        // 添加一些颜色变化
        cv::rectangle(image, cv::Point(50, 50), cv::Point(150, 150),
                      cv::Scalar(255, 0, 0), -1); // 蓝色区域
        cv::rectangle(image, cv::Point(200, 50), cv::Point(300, 150),
                      cv::Scalar(0, 255, 0), -1); // 绿色区域
        cv::rectangle(image, cv::Point(50, 200), cv::Point(150, 300),
                      cv::Scalar(0, 0, 255), -1); // 红色区域
    }

    // 1. 通道拆分
    std::vector<cv::Mat> channels;
    cv::split(image, channels); // 拆分为 B, G, R 三个单通道

    cv::Mat b_channel = channels[0]; // Blue
    cv::Mat g_channel = channels[1]; // Green
    cv::Mat r_channel = channels[2]; // Red

    // 显示各通道（灰度显示）
    cv::imshow("Original", image);
    cv::imshow("Blue Channel", b_channel);
    cv::imshow("Green Channel", g_channel);
    cv::imshow("Red Channel", r_channel);

    // 2. 将单通道转换为伪彩色显示（更直观）
    cv::Mat b_color, g_color, r_color;
    std::vector<cv::Mat> b_merge = {b_channel, cv::Mat::zeros(image.size(), CV_8UC1),
                                    cv::Mat::zeros(image.size(), CV_8UC1)};
    std::vector<cv::Mat> g_merge = {cv::Mat::zeros(image.size(), CV_8UC1),
                                    g_channel, cv::Mat::zeros(image.size(), CV_8UC1)};
    std::vector<cv::Mat> r_merge = {cv::Mat::zeros(image.size(), CV_8UC1),
                                    cv::Mat::zeros(image.size(), CV_8UC1),
                                    r_channel};

    cv::merge(b_merge, b_color); // 蓝色通道显示为蓝色
    cv::merge(g_merge, g_color); // 绿色通道显示为绿色
    cv::merge(r_merge, r_color); // 红色通道显示为红色

    cv::imshow("Blue as Color", b_color);
    cv::imshow("Green as Color", g_color);
    cv::imshow("Red as Color", r_color);

    // 3. 通道合并 - 恢复原图
    cv::Mat merged;
    cv::merge(channels, merged);
    cv::imshow("Merged Back", merged);

    // 4. 交换通道顺序（BGR -> RGB）
    std::vector<cv::Mat> rgb_channels = {r_channel, g_channel, b_channel};
    cv::Mat rgb_image;
    cv::merge(rgb_channels, rgb_image);
    cv::imshow("RGB Order", rgb_image);

    // 5. 只保留红色通道（将其他通道置零）
    std::vector<cv::Mat> red_only = {
        cv::Mat::zeros(image.size(), CV_8UC1),
        cv::Mat::zeros(image.size(), CV_8UC1),
        r_channel};
    cv::Mat red_filtered;
    cv::merge(red_only, red_filtered);
    cv::imshow("Red Only", red_filtered);

    cv::waitKey(0);
    return 0;
}
