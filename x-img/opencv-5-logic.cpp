#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 创建二值掩膜图像
    cv::Mat mask1 = cv::Mat::zeros(400, 400, CV_8UC1);
    cv::Mat mask2 = cv::Mat::zeros(400, 400, CV_8UC1);

    // 掩膜1：圆形
    cv::circle(mask1, cv::Point(150, 200), 100, cv::Scalar(255), -1);

    // 掩膜2：矩形
    cv::rectangle(mask2, cv::Point(200, 100), cv::Point(350, 300),
                  cv::Scalar(255), -1);

    // ===== 逻辑运算 =====

    // 1. 与运算（AND）- 交集
    cv::Mat bitwise_and;
    cv::bitwise_and(mask1, mask2, bitwise_and);

    // 2. 或运算（OR）- 并集
    cv::Mat bitwise_or;
    cv::bitwise_or(mask1, mask2, bitwise_or);

    // 3. 异或运算（XOR）- 对称差
    cv::Mat bitwise_xor;
    cv::bitwise_xor(mask1, mask2, bitwise_xor);

    // 4. 非运算（NOT）- 取反
    cv::Mat bitwise_not;
    cv::bitwise_not(mask1, bitwise_not);

    // ===== 实际应用：图像掩膜 =====

    // 创建彩色背景图
    cv::Mat background(400, 400, CV_8UC3, cv::Scalar(128, 128, 128));

    // 创建前景图（彩色圆形）
    cv::Mat foreground = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::circle(foreground, cv::Point(200, 200), 150,
               cv::Scalar(0, 255, 255), -1); // 黄色圆

    // 使用圆形掩膜提取前景
    cv::Mat mask_circle = cv::Mat::zeros(400, 400, CV_8UC1);
    cv::circle(mask_circle, cv::Point(200, 200), 150, cv::Scalar(255), -1);

    // 方法1: 使用 copyTo 和掩膜
    cv::Mat result1 = background.clone();
    foreground.copyTo(result1, mask_circle);

    // 方法2: 使用 bitwise 运算
    cv::Mat fg_masked, bg_masked, result2;
    cv::bitwise_and(foreground, foreground, fg_masked, mask_circle);
    cv::bitwise_and(background, background, bg_masked,
                    cv::Mat() /* empty mask means all */,
                    cv::bitwise_not(mask_circle));
    cv::add(fg_masked, bg_masked, result2);

    // 方法3: 更简洁的 bitwise 组合
    cv::Mat result3;
    cv::bitwise_or(background, cv::Scalar(0, 0, 0), result3,
                   cv::bitwise_not(mask_circle));
    cv::bitwise_or(result3, foreground, result3, mask_circle);

    // ===== 显示结果 =====
    cv::imshow("Mask 1 (Circle)", mask1);
    cv::imshow("Mask 2 (Rect)", mask2);
    cv::imshow("AND (Intersection)", bitwise_and);
    cv::imshow("OR (Union)", bitwise_or);
    cv::imshow("XOR (Symmetric Diff)", bitwise_xor);
    cv::imshow("NOT (Inverse)", bitwise_not);

    cv::imshow("Background", background);
    cv::imshow("Foreground", foreground);
    cv::imshow("Result (copyTo)", result1);
    cv::imshow("Result (bitwise)", result2);

    // ===== 高级应用：绿幕抠图 =====
    // 模拟绿幕图像
    cv::Mat greenscreen(400, 400, CV_8UC3, cv::Scalar(0, 255, 0)); // 绿色背景
    cv::circle(greenscreen, cv::Point(200, 200), 100,
               cv::Scalar(255, 0, 0), -1); // 蓝色圆

    // 提取非绿色区域
    cv::Mat hsv;
    cv::cvtColor(greenscreen, hsv, cv::COLOR_BGR2HSV);

    // HSV中绿色范围
    cv::Mat green_mask;
    cv::inRange(hsv, cv::Scalar(35, 40, 40), cv::Scalar(85, 255, 255), green_mask);
    cv::bitwise_not(green_mask, green_mask); // 反转：保留非绿色

    cv::Mat extracted;
    greenscreen.copyTo(extracted, green_mask);
    cv::imshow("Greenscreen", greenscreen);
    cv::imshow("Extracted (not green)", extracted);

    cv::waitKey(0);
    return 0;
}
