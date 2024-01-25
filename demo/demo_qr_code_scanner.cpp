/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

int demo_qr_code_scanner(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "main";
    const auto code_wnd = "code";

    cv::namedWindow(main_wnd);
    cv::namedWindow(code_wnd);

    cv::Mat frame;
    cv::Mat box;
    cv::Mat rect;

    cv::QRCodeDetector qrDecoder = cv::QRCodeDetector();

    const auto ESC_KEY_CODE = 27;
    while (cv::waitKey(30) != ESC_KEY_CODE)
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        std::string data = qrDecoder.detectAndDecode(frame, box, rect);
        if (data.length() > 0)
        {
            std::cout << "Data: " << data << std::endl;

            rect.convertTo(rect, CV_8UC3);

            cv::imshow(main_wnd, frame);
            cv::imshow(code_wnd, rect);
        }
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(code_wnd);

    return 0;
}
