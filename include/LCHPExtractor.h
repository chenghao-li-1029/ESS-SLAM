//
// Created by xddz on 19-5-24.
//

#ifndef LCHPEXTRACTOR_H
#define LCHPEXTRACTOR_H

#include <torch/script.h>
#include <torch/torch.h>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

#include <vector>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core/mat.hpp>

namespace ORB_SLAM2 {
    class LCHPExtractor {
    public:
        enum {
            HARRIS_SCORE = 0, FAST_SCORE = 1
        };

        struct lchp{
            lchp() = default;
            lchp(cv::Point2f pts_, int index_, float conf_):pts(pts_), index(index_), conf(conf_){}

            cv::Point2f pts;
            int index;
            float conf;
        };
        static bool GreaterSort (lchp a,lchp b) { return (a.conf>b.conf); }

        LCHPExtractor(int nfeatures_, float scaleFactor_, int nlevels_, int iniThFAST_, int minThFAST_);

        ~LCHPExtractor() {}

        void operator()(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors);

        void nms(cv::Mat det, cv::Mat desc, std::vector<cv::KeyPoint> &pts, cv::Mat &descriptors, int img_width, int img_height);

        void detectAndCompute(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

        cv::Mat preprocess(cv::Mat img);

        int inline GetLevels() {
            return nlevels;
        }

        float inline GetScaleFactor() {
            return scaleFactor;
        }

        std::vector<float> inline GetScaleFactors() {
            return mvScaleFactor;
        }

        std::vector<float> inline GetInverseScaleFactors() {
            return mvInvScaleFactor;
        }

        std::vector<float> inline GetScaleSigmaSquares() {
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares() {
            return mvInvLevelSigma2;
        }

        std::vector<cv::Mat> mvImagePyramid;

    protected:
        int nfeatures;
        float scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

//        int border = 8;
//        int dist_thresh = 4;
        int border = 4;
        int dist_thresh = 2;

        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

//        std::shared_ptr<torch::jit::script::Module> module;
        torch::jit::script::Module module;
        torch::Device device = torch::kCUDA;
    };

} //namespace ORB_SLAM2

#endif //LCHPEXTRACTOR_H
