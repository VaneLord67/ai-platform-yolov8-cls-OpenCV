#include <iostream>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp_ai_utils.h"

namespace py = pybind11;

extern const char* class_names[];

struct ClsResult {
    int label;
    std::string class_name;
    double confidence;
};

enum class InputStream { IMAGE, VIDEO, CAMERA };

ClsResult process_frame(cv::Mat& frame, cv::dnn::Net& net) {
    ClsResult clsResult{ 0 };

    if (frame.empty()) {
        std::cerr << "frame empty!" << std::endl;
        return clsResult;
    }

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), false, false);

    net.setInput(blob);
    cv::Mat output = net.forward();

    // 找到最大值及其位置
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc);

    // 输出最大值及其位置
    /*std::cout << "Max Value: " << maxVal << std::endl;
    std::cout << "Max Value Location: " << maxLoc << std::endl;
    std::cout << "class_name:" << class_names[maxLoc.x] << std::endl;*/
    clsResult.label = maxLoc.x;
    clsResult.confidence = maxVal;
    clsResult.class_name = class_names[maxLoc.x];

    return clsResult;
}

std::vector<ClsResult> main_func(int argc, char** argv) {
    std::vector<ClsResult> clsResults;

    cv::CommandLineParser parser(argc, argv,
        {
            "{video||video's path}"
            "{cam_id||camera's device id}"
            "{img||image's path}"
            "{queueName|| camera jpg data queue   }"
            "{stopSignalKey|| stop camera signal key  }"
            "{logKey||log key}"
            "{videoOutputPath||video output path}"
            "{videoProgressKey||video progress key}"
            "{videoOutputJsonPath||video output json path}"
        });

    std::string imagePath = "E:/GraduationDesign/tensorrt-alpha/data/sailboat3.jpg";
    std::string videoPath = "E:/GraduationDesign/tensorrt-alpha/data/people_h264.mp4";
    int cameraId = 0;
    std::string queueName = "";
    std::string stopSignalKey = "";
    std::string logKey = "";
    std::string videoOutputPath = "";
    std::string videoProgressKey = "";
    std::string videoOutputJsonPath = "";

    auto source = InputStream::IMAGE;

    std::cout << "parser args..." << std::endl;
    if (parser.has("img")) {
        imagePath = parser.get<std::string>("img");
        source = InputStream::IMAGE;
    }
    if (parser.has("video")) {
        videoPath = parser.get<std::string>("video");
        source = InputStream::VIDEO;
    }
    if (parser.has("cam_id")) {
        cameraId = parser.get<int>("cam_id");
        std::cout << "cam_id = " << cameraId << std::endl;
        source = InputStream::CAMERA;
    }
    if (parser.has("queueName")) {
        queueName = parser.get<std::string>("queueName");
        std::cout << "queueName = " << queueName << std::endl;
    }
    if (parser.has("stopSignalKey")) {
        stopSignalKey = parser.get<std::string>("stopSignalKey");
        std::cout << "stopSignalKey = " << stopSignalKey << std::endl;
    }
    if (parser.has("logKey")) {
        logKey = parser.get<std::string>("logKey");
        std::cout << "logKey = " << logKey << std::endl;
    }
    if (parser.has("videoOutputPath")) {
        videoOutputPath = parser.get<std::string>("videoOutputPath");
        std::cout << "videoOutputPath = " << videoOutputPath << std::endl;
    }
    if (parser.has("videoProgressKey")) {
        videoProgressKey = parser.get<std::string>("videoProgressKey");
        std::cout << "videoProgressKey = " << videoProgressKey << std::endl;
    }
    if (parser.has("videoOutputJsonPath")) {
        videoOutputJsonPath = parser.get<std::string>("videoOutputJsonPath");
        std::cout << "videoOutputJsonPath = " << videoOutputJsonPath << std::endl;
    }
    cpp_ai_utils::CppAiHelper cppAiHelper(logKey, queueName, stopSignalKey, 
        videoOutputPath, videoProgressKey, videoOutputJsonPath, videoPath);

    cv::VideoCapture capture;
    switch (source)
    {
    case InputStream::VIDEO:
        capture.open(videoPath);
        cppAiHelper.init_video_writer(capture);
        break;
    case InputStream::CAMERA:
        capture.open(cameraId);
        std::cout << "capture state: " << capture.isOpened() << std::endl;
        if (!capture.isOpened()) {
            cppAiHelper.push_log_to_redis(u8"打开摄像头失败！");
        }
        cppAiHelper.init_video_writer(capture);
        break;
    default:
        break;
    }

    std::cout << "model init..." << std::endl;

    std::string modelPath = "E:/GraduationDesign/yolov8n-cls.onnx";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    std::cout << "process frame..." << std::endl;

    cv::Mat image;
    cv::Mat frame;
    ClsResult tempResult;
    switch (source)
    {
    case InputStream::IMAGE:
        image = cv::imread(imagePath); // 读取图像
        if (image.empty()) {
            std::cerr << "Failed to read image!" << std::endl;
            cppAiHelper.push_log_to_redis("读取图像失败!");
            return clsResults;
        }
        clsResults.push_back(process_frame(image, net));
        break;
    case InputStream::VIDEO:
        while (capture.isOpened()) {
            capture.read(frame);
            if (frame.empty()) {
                break;
            }
            tempResult = process_frame(frame, net);
            // clsResults.push_back(tempResult);
            std::stringstream ss;
            ss << "{"
                << "\"class_name\":\"" << tempResult.class_name << "\","
                << "\"confidence\":" << tempResult.confidence << ","
                << "\"label\":" << tempResult.label << ""
                << "}";
            cppAiHelper.write_frame_to_video(frame);
            cppAiHelper.write_json_to_file(ss.str());
        }
        break;
    case InputStream::CAMERA:
        while (capture.isOpened()) {
            if (cppAiHelper.should_stop_camera()) {
                break;
            }
            capture.read(frame);
            if (frame.empty()) {
                break;
            }
            tempResult = process_frame(frame, net);
            // camera_stream process
            if (queueName != "") {
                cppAiHelper.push_frame_to_redis(frame);
                cppAiHelper.write_frame_to_video(frame);

                std::stringstream ss;
                ss << "{"
                    << "\"class_name\":\"" << tempResult.class_name << "\","
                    << "\"confidence\":" << tempResult.confidence << ","
                    << "\"label\":" << tempResult.label << ""
                    << "}";

                cppAiHelper.push_str_to_redis(ss.str());
                cppAiHelper.write_json_to_file(ss.str());
            }
        }
        break;
    default:
        break;
    }
    //cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    //cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), false, false);

    //net.setInput(blob);
    //cv::Mat output = net.forward();

    //// 找到最大值及其位置
    //double minVal, maxVal;
    //cv::Point minLoc, maxLoc;
    //cv::minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc);

    //// 输出最大值及其位置
    ///*std::cout << "Max Value: " << maxVal << std::endl;
    //std::cout << "Max Value Location: " << maxLoc << std::endl;
    //std::cout << "class_name:" << class_names[maxLoc.x] << std::endl;*/
    //clsResult.label = maxLoc.x;
    //clsResult.confidence = maxVal;
    //clsResult.class_name = class_names[maxLoc.x];

    return clsResults;
}

std::vector<ClsResult> main_func_wrapper(const std::vector<std::string>& strings) {
    int argc = static_cast<int>(strings.size());
    std::vector<char*> cstrings;
    cstrings.reserve(strings.size());
    for (size_t i = 0; i < strings.size(); ++i) {
        cstrings.push_back(const_cast<char*>(strings[i].c_str()));
    }
    auto r = main_func(argc, &cstrings[0]);
    return r;
}

#ifdef _WIN32
PYBIND11_MODULE(app_yolo_cls, m) {
#else
PYBIND11_MODULE(libapp_yolo_cls, m) {
#endif
    m.doc() = "pybind11 example plugin"; // optional module docstring

    //m.def("add", &add, "A function that adds two numbers");
    py::class_<ClsResult>(m, "ClsResult")
        .def_readwrite("confidence", &ClsResult::confidence)
        .def_readwrite("class_name", &ClsResult::class_name)
        .def_readwrite("label", &ClsResult::label);

    m.def("main_func_wrapper", &main_func_wrapper, "main func");
}
