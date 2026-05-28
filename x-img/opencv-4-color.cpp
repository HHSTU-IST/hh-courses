#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 创建测试图像（BGR格式）
    cv::Mat bgr_image(300, 600, CV_8UC3);
    // 左半部分蓝色 (255,0,0 in BGR), 右半部分红色 (0,0,255 in BGR)
    cv::Mat left_half = bgr_image(cv::Rect(0, 0, 300, 300));
    cv::Mat right_half = bgr_image(cv::Rect(300, 0, 300, 300));
    left_half.setTo(cv::Scalar(255, 0, 0));  // 纯蓝 (BGR)
    right_half.setTo(cv::Scalar(0, 0, 255)); // 纯红 (BGR)

    // 方法1: 使用 cv::cvtColor 进行 BGR <-> RGB 转换
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    // 方法2: 手动交换通道（使用split/merge）
    std::vector<cv::Mat> channels;
    cv::split(bgr_image, channels);
    std::swap(channels[0], channels[2]); // 交换B和R
    cv::Mat rgb_manual;
    cv::merge(channels, rgb_manual);

    // 方法3: 使用 mixChannels 进行通道映射
    // 从src的 [0,1,2] 通道映射到dst的 [2,1,0] 通道（即BGR->RGB）
    cv::Mat rgb_mix(bgr_image.size(), CV_8UC3);
    int from_to[] = {0, 2, 1, 1, 2, 0}; // src0->dst2, src1->dst1, src2->dst0
    cv::mixChannels(&bgr_image, 1, &rgb_mix, 1, from_to, 3);

    // 显示对比
    cv::imshow("Original BGR", bgr_image);
    cv::imshow("RGB (cvtColor)", rgb_image);
    cv::imshow("RGB (Manual)", rgb_manual);
    cv::imshow("RGB (mixChannels)", rgb_mix);

    // 验证：RGB转回BGR
    cv::Mat bgr_back;
    cv::cvtColor(rgb_image, bgr_back, cv::COLOR_RGB2BGR);
    cv::imshow("BGR Converted Back", bgr_back);

    // 检查是否与原图相同
    cv::Mat diff;
    cv::absdiff(bgr_image, bgr_back, diff);
    double max_diff = cv::norm(diff, cv::NORM_INF);
    std::cout << "Max difference after round-trip: " << max_diff << std::endl;

    cv::waitKey(0);
    return 0;
}
