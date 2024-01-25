/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2023-11-14
 * @author Yaroslav Murenkov
 */

#include <utility>
#include <vector>

#include "cvlib.hpp"

namespace
{
struct QTree
{
    cv::Mat data;
    std::vector<QTree> children;
    bool has_children;

    QTree(cv::Mat data) : data(data), has_children(false)
    {
    }
};

void split_image(cv::Mat image, QTree* tree, double stddev)
{
    cv::Mat mean;
    cv::Mat dev;

    const auto width = image.cols;
    const auto height = image.rows;

    if (width == 1 || height == 1)
        return;

    cv::meanStdDev(image, mean, dev);
    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }

    tree->data = image;
    tree->has_children = true;

    cv::Mat north_west = image(cv::Range(0, height / 2), cv::Range(0, width / 2));
    cv::Mat north_east = image(cv::Range(0, height / 2), cv::Range(width / 2, width));
    cv::Mat south_west = image(cv::Range(height / 2, height), cv::Range(0, width / 2));
    cv::Mat south_east = image(cv::Range(height / 2, height), cv::Range(width / 2, width));

    size_t i = 0;
    for (const auto& child : {north_west, north_east, south_west, south_east})
    {
        tree->children.push_back(QTree(child));
        split_image(child, &tree->children[i++], stddev);
    }
}

void merge_image(QTree* tree, double stddev)
{
    auto merge_children = [](std::pair<QTree*, QTree*> children, double stddev) {
        cv::Mat mean1, dev1;
        cv::Mat mean2, dev2;

        cv::meanStdDev(children.first->data, mean1, dev1);
        cv::meanStdDev(children.second->data, mean2, dev2);

        cv::Mat mean = (mean1 + mean2) / 2;
        if ((dev1.at<double>(0) < stddev) && (dev2.at<double>(0) < stddev))
        {
            children.first->data.setTo(mean);
            children.second->data.setTo(mean);
        }
    };

    while (tree->has_children)
    {
        bool has_child = false;
        for (const auto& child : tree->children)
            has_child |= child.has_children;

        if (has_child)
        {
            for (auto& child : tree->children)
                merge_image(&child, stddev);
        }
        else
        {
            for (auto& pair : {std::make_pair(&tree->children[0], &tree->children[1]), std::make_pair(&tree->children[0], &tree->children[2]),
                               std::make_pair(&tree->children[1], &tree->children[3]), std::make_pair(&tree->children[2], &tree->children[3])})
                merge_children(pair, stddev);
            tree->has_children = false;
        }
    }
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    cv::Mat res = image;
    QTree tree(res);

    split_image(res, &tree, stddev);
    merge_image(&tree, stddev);

    return res;
}
} // namespace cvlib
