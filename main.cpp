#include <torch/torch.h>

#include<ANN.h>

int main() {
  auto ann = std::make_shared<ANN>();

  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./data").map(
          torch::data::transforms::Stack<>()),
      /*batch_size=*/50);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::Adam optimizer(ann->parameters(), /*lr=*/0.001);

  for (size_t epoch = 1; epoch <= 20; ++epoch) {
    size_t batch_index = 0;
    for (auto & batch : *data_loader) {
      optimizer.zero_grad();
      torch::Tensor prediction = ann->forward(batch.data);
      torch::Tensor loss = torch::nn::functional::cross_entropy(prediction, batch.target);
      loss.backward();
      optimizer.step();
      if (++batch_index % 50 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(ann, "libtorch/ann.pt");
      }
    }
  }
