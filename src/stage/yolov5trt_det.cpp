#include "stage/yolov5trt_det.h"

Yolov5TrtDet::Yolov5TrtDet(std::string engine) {
    engine_name_ = engine;

    cudaSetDevice(kGpuId);

    if (!wts_name_.empty()) {
        SerializeEngine(kBatchSize, is_p6_, gd_, gw_, wts_name_, engine_name_);
    }

    DeserializeEngine(engine_name_, &runtime_, &engine_, &context_);

    CUDA_CHECK(cudaStreamCreate(&stream_));

    // 初始化cuda预处理
    CudaPreprocessInit(kMaxInputImageSize);

    // 准备CPU和GPU缓冲区
    PrepareBuffers(engine_, &gpu_buffers_[0], &gpu_buffers_[1],
                   &cpu_output_buffer_);
}

Yolov5TrtDet::~Yolov5TrtDet() {
    // 释放流和缓冲区
    cudaStreamDestroy(stream_);
    CUDA_CHECK(cudaFree(gpu_buffers_[0]));
    CUDA_CHECK(cudaFree(gpu_buffers_[1]));
    delete[] cpu_output_buffer_;
    CudaPreprocessDestroy();
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}

/**
 * @brief 预处理缓冲区
 *
 * @param engine 引擎
 * @param gpu_input_buffer GPU输入缓冲区
 * @param gpu_output_buffer GPU输出缓冲区
 * @param cpu_output_buffer CPU输出缓冲区
 */
void Yolov5TrtDet::PrepareBuffers(ICudaEngine* engine, float** gpu_input_buffer,
                                  float** gpu_output_buffer,
                                  float** cpu_output_buffer) {
    assert(engine->getNbBindings() == 2);
    // 为了绑定缓冲区，需要知道输入张量和输出张量的名称。注意，保证索引值小于IEngine::getNbBindings()
    const int inputIndex  = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // 创建GPU缓冲区
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer,
                          kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer,
                          kBatchSize * kOutputSize * sizeof(float)));

    *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

/**
 * @brief 推理
 *
 * @param context 推理环境
 * @param stream cuda流
 * @param gpu_buffers GPU缓冲区
 * @param output 结果输出
 * @param batchsize batch size
 */
void Yolov5TrtDet::Infer(IExecutionContext& context, cudaStream_t& stream,
                         void** gpu_buffers, float* output, int batchsize) {
    context.enqueue(batchsize, gpu_buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1],
                               batchsize * kOutputSize * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

/**
 * @brief 序列化引擎
 *
 * @param max_batchsize 最大batch size
 * @param is_p6 是否使用p6模型
 * @param gd 检测阈值
 * @param gw 追踪阈值
 * @param wts_name 权重
 * @param engine_name 引擎
 */
void Yolov5TrtDet::SerializeEngine(unsigned int max_batchsize, bool is_p6,
                                   float gd, float gw, std::string wts_name,
                                   std::string engine_name) {
    // 创建构建器
    IBuilder* builder      = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // 创建模型来填充网络，然后设置输出并创建引擎
    ICudaEngine* engine = nullptr;
    if (is_p6) {
        engine = BuildDetP6Engine(max_batchsize, builder, config,
                                  DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = BuildDetEngine(max_batchsize, builder, config,
                                DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // 序列化引擎
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // 保存引擎到文件中
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()),
            serialized_engine->size());

    // 关闭引擎、构建器、配置器
    engine->destroy();
    builder->destroy();
    config->destroy();
    serialized_engine->destroy();
}

/**
 * @brief 反序列化引擎
 *
 * @param engine_name 引擎
 * @param runtime runtime实例
 * @param engine 引擎
 * @param context 推理环境
 */
void Yolov5TrtDet::DeserializeEngine(std::string& engine_name,
                                     IRuntime** runtime, ICudaEngine** engine,
                                     IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

/**
 * @brief 图像检测
 *
 * @param imgs 源数据
 */
void Yolov5TrtDet::Detect(std::vector<cv::Mat> imgs,
                          std::vector<cv::Mat>& result_imgs) {
    std::vector<std::string> file_names;
    if (USE_IMG_DIR) {
        // 从目录获取图像
        if (readFilesInDir(img_dir_.c_str(), file_names) < 0) {
            std::cerr << "read_files_in_dir failed." << std::endl;
        }
    } else {
        file_names.resize(imgs.size());
    }

    // 批处理预测
    for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
        // 获取批处理数据
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;

        if (USE_IMG_DIR) {
            for (size_t j = i; j < i + kBatchSize && j < file_names.size();
                 j++) {
                cv::Mat img = cv::imread(img_dir_ + "/" + file_names[j]);
                img_batch.push_back(img);
                img_name_batch.push_back(file_names[j]);
            }
        } else {
            for (size_t i = 0; i < imgs.size(); i++) {
                img_batch.push_back(imgs[i]);
            }
        }

        // 预处理
        CudaBatchPreprocess(img_batch, gpu_buffers_[0], kInputW, kInputH,
                            stream_);

        // 推理
        auto start = std::chrono::system_clock::now();
        Infer(*context_, stream_, (void**)gpu_buffers_, cpu_output_buffer_,
              kBatchSize);
        auto end = std::chrono::system_clock::now();
        LOG_F(INFO, "inference time: %d ms",
              std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count());

        // NMS
        std::vector<std::vector<Detection>> res_batch;
        BatcNms(res_batch, cpu_output_buffer_, img_batch.size(), kOutputSize,
                kConfThresh, kNmsThresh);

        // 画框
        DrawBbox(img_batch, res_batch);

        // 保存结果
        for (size_t j = 0; j < img_batch.size(); j++) {
            // cv::imwrite("_" + img_name_batch[j], img_batch[j]);
            result_imgs.push_back(img_batch[j]);
        }
    }
}

void Yolov5TrtDet::DetectImgDir(std::string dir) {
    std::vector<std::string> file_names;

    // 从目录获取图像
    if (readFilesInDir(dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
    }

    // 批处理预测
    for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
        // 获取批处理数据
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;

        for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
            cv::Mat img = cv::imread(dir + "/" + file_names[j]);
            img_batch.push_back(img);
            img_name_batch.push_back(file_names[j]);
        }

        // 预处理
        CudaBatchPreprocess(img_batch, gpu_buffers_[0], kInputW, kInputH,
                            stream_);

        // 推理
        auto start = std::chrono::system_clock::now();
        Infer(*context_, stream_, (void**)gpu_buffers_, cpu_output_buffer_,
              kBatchSize);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - start)
                         .count()
                  << "ms" << std::endl;

        // NMS
        std::vector<std::vector<Detection>> res_batch;
        BatcNms(res_batch, cpu_output_buffer_, img_batch.size(), kOutputSize,
                kConfThresh, kNmsThresh);

        // 画框
        DrawBbox(img_batch, res_batch);

        // 保存结果
        for (size_t j = 0; j < img_batch.size(); j++) {
            cv::imwrite("_" + img_name_batch[j], img_batch[j]);
            cv::imshow("_" + img_name_batch[j], img_batch[j]);
        }
    }
}

void Yolov5TrtDet::DetectCamera(cv::Mat img, cv::Mat& result_img) {
    // 获取批处理数据
    std::vector<cv::Mat> img_batch;
    img_batch.push_back(img);

    // 预处理
    CudaBatchPreprocess(img_batch, gpu_buffers_[0], kInputW, kInputH, stream_);

    // 推理
    auto start = std::chrono::system_clock::now();
    Infer(*context_, stream_, (void**)gpu_buffers_, cpu_output_buffer_,
          kBatchSize);
    auto end = std::chrono::system_clock::now();
    LOG_F(INFO, "inference time: %d ms",
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count());

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    BatcNms(res_batch, cpu_output_buffer_, img_batch.size(), kOutputSize,
            kConfThresh, kNmsThresh);

    // 画框
    DrawBbox(img_batch, res_batch);

    // 保存结果
    result_img = img_batch[0];
}

struct struct_yolo_result Yolov5TrtDet::detect_bbox(cv::Mat img) {
    struct struct_yolo_result result;

    // 获取批处理数据
    std::vector<cv::Mat> img_batch;
    img_batch.push_back(img);

    // 预处理
    CudaBatchPreprocess(img_batch, gpu_buffers_[0], kInputW, kInputH, stream_);

    // 推理
    auto start = std::chrono::system_clock::now();
    Infer(*context_, stream_, (void**)gpu_buffers_, cpu_output_buffer_,
          kBatchSize);
    auto end = std::chrono::system_clock::now();
    LOG_F(INFO, "inference time: %d ms",
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count());

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    BatcNms(res_batch, cpu_output_buffer_, img_batch.size(), kOutputSize,
            kConfThresh, kNmsThresh);

    // 画框
    // DrawBbox(img_batch, res_batch);
    DrawBboxWithResult(img_batch, res_batch, result);

    // 保存结果
    // result_img = img_batch[0];
    return result;
}