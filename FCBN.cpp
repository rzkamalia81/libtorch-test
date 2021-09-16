%writefile FCBN.cpp

FCBN::FCBN(int in_features, int out_features){
    ln = register_module("ln", torch::nn::Linear(in_features, out_features));
    bn = register_module("bn", torch::nn::BatchNorm1d(out_features));
    }

torch::Tensor FCBN::forward(torch::Tensor x){
    x = ln->forward(x);
    x = bn(x);
    return x;
    }
