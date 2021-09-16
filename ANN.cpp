%%writefile ANN.cpp

#include<ANN.h>

ANN::ANN(){
    input = FCBN(node[0], nodes[1]);
    hidden = FCBN(nodes[1], nodes[2]);
    output = FCBN(nodes[2], nodes[3]);

    input = register_module("input", input);
    hidden = register_module("hidden", hidden);
    output = register_module("output", output);
    }

torch::Tensor ANN::forward(torch::Tensor x){
    x = input->forward(x);
    x = hidden->forward(x);
    x = output->forward(x);
    return x;
    }
