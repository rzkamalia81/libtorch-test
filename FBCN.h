%%writefile FCBN.h

class FCBN : public torch::nn::Module{
    public:
    FCBN(int intput_features, int output_features);
    torch::Tensor forward(torch::Tensor x);
    private:
    torch::nn::Linear ln{nullptr};
    torch::nn::BatchNorm1d bn{nullptr};
    };
    TORCH_MODULE(FCBN);
