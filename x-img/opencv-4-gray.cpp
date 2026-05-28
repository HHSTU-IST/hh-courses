#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 读取或创建彩色图像
    cv::Mat color_img = cv::imread("lena.jpg");
    if (color_img.empty())
    {
        color_img = cv::Mat(400, 400, CV_8UC3);
        // 创建彩色渐变
        for (int i = 0; i < 400; i++)
        {
            for (int j = 0; j < 400; j++)
            {
                color_img.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    j % 255,       // B
                    (i + j) % 255, // G
                    i % 255        // R
                );
            }
        }
    }

    // 方法1: cvtColor 彩色转灰度
    cv::Mat gray1;
    cv::cvtColor(color_img, gray1, cv::COLOR_BGR2GRAY);

    // 方法2: 手动计算（加权平均：Y = 0.299R + 0.587G + 0.114B）
    cv::Mat gray2(color_img.size(), CV_8UC1);
    for (int i = 0; i < color_img.rows; i++)
    {
        for (int j = 0; j < color_img.cols; j++)
        {
            cv::Vec3b pixel = color_img.at<cv::Vec3b>(i, j);
            // B=pixel[0], G=pixel[1], R=pixel[2]
            double gray_val = 0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2];
            gray2.at<uchar>(i, j) = cv::saturate_cast<uchar>(gray_val);
        }
    }

    // 方法3: 使用 convertTo 或矩阵运算（更高效）
    cv::Mat gray3;
    // 创建转换矩阵: [0.114, 0.587, 0.299] 对应 [B, G, R]
    cv::Mat transform = (cv::Mat_<float>(1, 3) << 0.114, 0.587, 0.299);
    // 需要reshape图像为像素列表后做矩阵乘法，这里简化展示
    // 实际使用 cvtColor 即可

    // 方法4: 直接取某一通道（非标准灰度，仅作对比）
    std::vector<cv::Mat> channels;
    cv::split(color_img, channels);
    cv::Mat gray_b = channels[0]; // 仅蓝色通道
    cv::Mat gray_g = channels[1]; // 仅绿色通道
    cv::Mat gray_r = channels[2]; // 仅红色通道

    // 显示所有结果
    cv::imshow("Original Color", color_img);
    cv::imshow("Gray (cvtColor)", gray1);
    cv::imshow("Gray (Manual)", gray2);
    cv::imshow("Blue Channel Only", gray_b);
    cv::imshow("Green Channel Only", gray_g);
    cv::imshow("Red Channel Only", gray_r);

    // 比较方法1和方法2的差异
    cv::Mat diff;
    cv::absdiff(gray1, gray2, diff);
    std::cout << "Difference between cvtColor and manual: "
              << cv::sum(diff)[0] << std::endl;

    // 灰度转回伪彩色（热力图效果）
    cv::Mat color_mapped;
    cv::applyColorMap(gray1, color_mapped, cv::COLORMAP_JET);
    cv::imshow("Pseudo Color (JET)", color_mapped);

    cv::waitKey(0);
    return 0;
}
