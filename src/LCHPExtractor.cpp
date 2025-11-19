//
// Created by xddz on 19-5-24.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "LCHPExtractor.h"

using namespace std;

namespace ORB_SLAM2 {
    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    void LCHPExtractor::nms(cv::Mat det, cv::Mat desc, vector<cv::KeyPoint> &pts, cv::Mat &descriptors,
            int img_width, int img_height) {
        vector<cv::Point2f> pts_raw;
        vector<float> confidence_raw;

        cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
        cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

        grid.setTo(0);
        inds.setTo(0);

        for (int i = 0; i < det.rows; i++) {
            int u = (int) det.at<float>(i, 0);
            int v = (int) det.at<float>(i, 1);
            float conf = det.at<float>(i, 2);

            pts_raw.push_back(cv::Point2f(u, v));
            confidence_raw.push_back(conf);

            grid.at<u_char>(v, u) = 1;
            inds.at<unsigned short>(v, u) = i;
        }
        // pad the border of the grid, so that we can NMS points near the border
        cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

        for (int i = 0; i < pts_raw.size(); i++) {
            int uu = (int) pts_raw[i].x + dist_thresh;
            int vv = (int) pts_raw[i].y + dist_thresh;

            if (grid.at<u_char>(vv, uu) != 1)
                continue;

            for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
                for (int j = -dist_thresh; j < (dist_thresh + 1); j++) {
                    if (j == 0 && k == 0) continue;

                    grid.at<u_char>(vv + k, uu + j) = 0;

                }
            grid.at<u_char>(vv, uu) = 2;
        }

        size_t valid_cnt = 0;
        vector<lchp> lchp_pts;
        vector<int> select_indice;

        for (int v = 0; v < (img_height + dist_thresh); v++) {
            for (int u = 0; u < (img_width + dist_thresh); u++) {
                if (u - dist_thresh >= (img_width - border) || u - dist_thresh < border ||
                    v - dist_thresh >= (img_height - border) || v - dist_thresh < border)
                    continue;

                if (grid.at<u_char>(v, u) == 2) {
                    int select_ind = (int) inds.at<unsigned short>(v - dist_thresh, u - dist_thresh);
                    lchp_pts.push_back(lchp(pts_raw[select_ind], select_ind, confidence_raw[select_ind]));
//                    pts.push_back(cv::KeyPoint(pts_raw[select_ind], 1.0f));
//                    select_indice.push_back(select_ind);
                    valid_cnt++;
                }
            }
        }

        sort(lchp_pts.begin(), lchp_pts.end(), GreaterSort);
        int max = ((int)lchp_pts.size() > nfeatures)? nfeatures: (int)lchp_pts.size();
        for(int i = 0; i < max; i ++){
            pts.push_back(cv::KeyPoint(lchp_pts[i].pts, 1.0f));
            select_indice.push_back((lchp_pts[i].index));
        }

        descriptors.create(select_indice.size(), 32, CV_8U); // pts length matches descriptors length
//        descriptors.create(select_indice.size(), 32, CV_32F); // pts length matches descriptors length

        for (int i = 0; i < select_indice.size(); i++) {
            for (int j = 0; j < 32; j++) {
                descriptors.at<u_char>(i, j) = desc.at<u_char>(select_indice[i], j);
//            descriptors.at<float_t >(i, j) = desc.at<float_t>(select_indice[i], j);
            }
        }
    }

    LCHPExtractor::LCHPExtractor(int nfeatures_, float scaleFactor_, int nlevels_, int iniThFAST_, int minThFAST_) :
            nfeatures(nfeatures_), scaleFactor(scaleFactor_), nlevels(nlevels_), iniThFAST(iniThFAST_),
            minThFAST(minThFAST_) {

        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0] = 1.0f;
        mvLevelSigma2[0] = 1.0f;
        for (int i = 1; i < nlevels; i++) {
            mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
            mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale =
                nfeatures * (1 - factor) / (1 - (float) pow((double) factor, (double) nlevels));

        int sumFeatures = 0;
        for (int level = 0; level < nlevels - 1; level++) {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }

        const char *net_fn = getenv("LCHP_PATH");
        net_fn = (net_fn == nullptr) ? "../model/lchpnet_v2_640_480_cpp_demo.pt" : net_fn;
        module = torch::jit::load(net_fn);
        module.to(device);

    }

    void LCHPExtractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                                   std::vector<cv::KeyPoint> &_keypoints, cv::OutputArray _descriptors) {
        if (_image.empty())
            return;

        cv::Mat image = _image.getMat();
        assert(image.type() == CV_8UC1);

        cv::Mat img;
        image.convertTo(img, CV_32FC1, 1.f / 255.f, 0);

        int img_rows = 480, img_cols = 640;
//        auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img.data, {1, img_rows, img_cols, 1});
        auto img_tensor = torch::from_blob(img.data, {1, img_rows, img_cols, 1});
        img_tensor = img_tensor.permute({0, 3, 1, 2});
        auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_var);
        auto output = module.forward(inputs).toTuple();

        auto pts = output->elements()[0].toTensor().to(torch::kCPU).squeeze();
        auto desc = output->elements()[1].toTensor().to(torch::kCPU).squeeze();

        auto pts_data = pts.data<float>();
        auto desc_data = desc.data<float>();

        cv::Mat pts_mat(cv::Size(pts.size(1), 3), CV_32FC1, pts_data);
        cv::Mat desc_raw(cv::Size(pts.size(1), 32), CV_32FC1, desc_data);
        cv::Mat desc_mat;

//        desc_raw.convertTo(desc_mat, CV_8U, 255.f, 128.f);
        desc_raw.convertTo(desc_mat, CV_8U, 183.f, 110.f);
//        desc_raw.convertTo(desc_mat, CV_32F, 255.f);

        cv::transpose(pts_mat, pts_mat);
        cv::transpose(desc_mat, desc_mat);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        nms(pts_mat, desc_mat, keypoints, descriptors, img_cols, img_rows);

        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());

        _descriptors.create((int) keypoints.size(), 32, CV_8U);
//        _descriptors.create((int) keypoints.size(), 32, CV_32F);
        descriptors.copyTo(_descriptors.getMat());
    }

    void LCHPExtractor::detectAndCompute(cv::Mat &image, vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
        cv::Mat src = preprocess(image);

        int img_rows = 480, img_cols = 640;
//        auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(src.data, {1, img_rows, img_cols, 1});
        auto img_tensor = torch::from_blob(src.data, {1, img_rows, img_cols, 1});
        img_tensor = img_tensor.permute({0, 3, 1, 2});
        auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_var);
        auto output = module.forward(inputs).toTuple();

        auto pts = output->elements()[0].toTensor().to(torch::kCPU).squeeze();
        auto desc = output->elements()[1].toTensor().to(torch::kCPU).squeeze();

        auto pts_data = pts.data<float>();
        auto desc_data = desc.data<float>();

        cv::Mat pts_mat(cv::Size(pts.size(1), 3), CV_32FC1, pts_data);
        cv::Mat desc_raw(cv::Size(pts.size(1), 32), CV_32FC1, desc_data);
        cv::Mat desc_mat;

//    desc_raw.convertTo(desc_mat, CV_8U, 128.f, 64.f);
//        desc_raw.convertTo(desc_mat, CV_8U, 255.f, 128.f);
        desc_raw.convertTo(desc_mat, CV_8U, 183.f, 110.f);

        cv::transpose(pts_mat, pts_mat);
        cv::transpose(desc_mat, desc_mat);

        nms(pts_mat, desc_mat, keypoints, descriptors, img_cols, img_rows);

    }

    cv::Mat LCHPExtractor::preprocess(cv::Mat img) {
//    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        cv::Mat src;
        assert(img.type() == CV_8UC1);
        img.convertTo(src, CV_32FC1, 1.f / 255.f, 0);
        return src;
    }

} // namespace ORB_SLAM2