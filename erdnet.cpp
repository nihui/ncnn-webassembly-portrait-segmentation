// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "erdnet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

int ERDNet::load(bool use_gpu)
{
    erdnet.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    erdnet.opt = ncnn::Option();

#if NCNN_VULKAN
    erdnet.opt.use_vulkan_compute = use_gpu;
#endif

    erdnet.opt.num_threads = ncnn::get_big_cpu_count();

    erdnet.load_param("erdnet.param");
    erdnet.load_model("erdnet.bin");

    return 0;
}

int ERDNet::detect(const cv::Mat& rgba, cv::Mat& mask_g)
{
    const int w = rgba.cols;
    const int h = rgba.rows;
    const int input_size = 256;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgba.data, ncnn::Mat::PIXEL_RGBA2BGR, w, h, input_size, input_size);

    const float mean_vals[3] = {104.f, 112.f, 121.f};
    const float norm_vals[3] = {1.f/255.f, 1.f/255.f, 1.f/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = erdnet.create_extractor();

    ex.input("input_blob1", in);

    ncnn::Mat out;
    ex.extract("sigmoid_blob1", out);

    const float denorm_vals[3] = {255.f, 255.f, 255.f};
    out.substract_mean_normalize(0, denorm_vals);

    mask_g.create(h, w, CV_8UC1);
    out.to_pixels_resize(mask_g.data, ncnn::Mat::PIXEL_GRAY, w, h);

    return 0;
}

int ERDNet::draw(cv::Mat& rgba, const cv::Mat& bg_bgr, const cv::Mat& mask_g)
{
    const int w = rgba.cols;
    const int h = rgba.rows;

    for (int y = 0; y < h; y++)
    {
        uchar* rgbaptr = rgba.ptr<uchar>(y);
        const uchar* bgptr = bg_bgr.ptr<const uchar>(y);
        const uchar* mptr = mask_g.ptr<const uchar>(y);

        for (int x = 0; x < w; x++)
        {
            const uchar alpha = mptr[0];

            rgbaptr[0] = cv::saturate_cast<uchar>((rgbaptr[0] * alpha + bgptr[2] * (255 - alpha)) / 255);
            rgbaptr[1] = cv::saturate_cast<uchar>((rgbaptr[1] * alpha + bgptr[1] * (255 - alpha)) / 255);
            rgbaptr[2] = cv::saturate_cast<uchar>((rgbaptr[2] * alpha + bgptr[0] * (255 - alpha)) / 255);

            rgbaptr += 4;
            bgptr += 3;
            mptr += 1;
        }
    }

    return 0;
}
