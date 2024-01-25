/* Split and merge segmentation algorithm testing.
 * @file
 * @date 2023-11-14
 * @author Yaroslav Murenkov
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("constant image", "[split_and_merge]")
{
    const cv::Mat image(100, 100, CV_8UC1, cv::Scalar{15});

    const auto res = split_and_merge(image, 1);
    REQUIRE(image.size() == res.size());
    REQUIRE(image.type() == res.type());
    REQUIRE(cv::Scalar(15) == cv::mean(res));
}

TEST_CASE("simple regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(2, 2) <<
                0, 1,
                2, 3
        );
        const cv::Mat reference = (cv::Mat_<char>(2, 2) <<
                2, 2,
                2, 2
        );
        // clang-format on
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("3x3")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(3, 3) <<
                0, 1, 2,
                3, 4, 5,
                6, 7, 8
        );
        const cv::Mat reference = (cv::Mat_<char>(3, 3) <<
                4, 4, 4,
                4, 4, 4,
                4, 4, 4
        );
        // clang-format on
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }
}

TEST_CASE("complex regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(2, 2) <<
                5, 7,
                6, 6
        );
        const cv::Mat reference = (cv::Mat_<char>(2, 2) <<
                6, 6,
                6, 6
        );
        // clang-format on
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("3x3")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(3, 3) <<
                40,  4,  4,
                40,  4,  4,
                40, 40, 40
        );
        const cv::Mat reference = (cv::Mat_<char>(3, 3) <<
                31, 22, 22,
                26, 26, 26,
                26, 26, 26
        );
        // clang-format on
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("4x4")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(4, 4) <<
                40, 43, 0, 3,
                41, 42, 6, 1,
                98, 92, 4, 2,
                96, 94, 5, 7
        );
        const cv::Mat reference = (cv::Mat_<char>(4, 4) <<
                58, 58, 13, 13,
                58, 58, 13, 13,
                36, 36, 36, 36,
                36, 36, 36, 36
        );
        // clang-format on
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }
}
