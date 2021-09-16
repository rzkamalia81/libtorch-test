%%writefile ANN.h

#include<FCBN.h>

class ANN: public torch::nn::Module{
    public:
    ANN();
    torch::Tensor forward(torch::Tensor x);
    private:
    int nodes[4] = {784,100,50,10};
    FCBN input{nullptr};
    FCBN hidden{nullptr};
    FCBN output{nullptr};
    };
