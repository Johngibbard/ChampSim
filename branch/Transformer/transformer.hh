// transformer.h

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <functional>
#include <memory>
#include <vector>

#include <torch/torch.h>

// LayerNorm module
class LayerNormImpl : public torch::nn::Module
{
public:
  LayerNormImpl(int64_t features, double eps = 1e-6)
      : a_2(register_parameter("a_2", torch::ones(features))), b_2(register_parameter("b_2", torch::zeros(features))), eps(eps)
  {
  }

  torch::Tensor forward(torch::Tensor x)
  {
    auto mean = x.mean(-1, true);
    auto std = x.std(-1, true);
    return a_2 * (x - mean) / (std + eps) + b_2;
  }

private:
  torch::Tensor a_2, b_2;
  double eps;
};
TORCH_MODULE(LayerNorm);

// SublayerConnection module with residual connection and dropout
class SublayerConnectionImpl : public torch::nn::Module
{
public:
  SublayerConnectionImpl(int64_t size, double dropout) : size(size), LayerNorm(LayerNormOptions({size})), dropout(torch::nn::Dropout(dropout))
  {
    register_module("norm", norm);
    register_module("dropout", dropout);
  }

  torch::Tensor forward(torch::Tensor x, const std::function<torch::Tensor(torch::Tensor)> sublayer) { return x + dropout(sublayer(norm->forward(x))); }

  std::shared_ptr<SublayerConnectionImpl> clone() const override { return std::make_shared<SublayerConnectionImpl>(size, dropout->p()); }

private:
  int64_t size;
  LayerNorm norm{nullptr};
  torch::nn::Dropout dropout;
};
TORCH_MODULE(SublayerConnection);

// Function to create N clones of a module
template <typename ModuleType>
std::vector<std::shared_ptr<ModuleType>> clones(const std::shared_ptr<ModuleType>& module, int64_t N)
{
  std::vector<std::shared_ptr<ModuleType>> modules;
  for (int64_t i = 0; i < N; ++i) {
    modules.push_back(std::dynamic_pointer_cast<ModuleType>(module->clone()));
  }
  return modules;
}

// EncoderLayer
class EncoderLayerImpl : public torch::nn::Module
{
public:
  EncoderLayerImpl(int64_t size, torch::nn::MultiheadAttention self_attn, torch::nn::Sequential feed_forward, double dropout)
      : self_attn(self_attn), feed_forward(feed_forward), sublayer(clones<SublayerConnection>(std::make_shared<SublayerConnection>(size, dropout), 2)),
        size(size)
  {
    register_module("self_attn", self_attn);
    register_module("feed_forward", feed_forward);
    for (size_t i = 0; i < sublayer.size(); ++i) {
      register_module("sublayer_" + std::to_string(i), sublayer[i]);
    }
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask)
  {
    x = sublayer[0]->forward(x, [&](torch::Tensor x) { return std::get<0>(self_attn->forward(x, x, x, mask)); });
    x = sublayer[1]->forward(x, feed_forward);
    return x;
  }

private:
  torch::nn::MultiheadAttention self_attn;
  torch::nn::Sequential feed_forward;
  std::vector<SublayerConnection> sublayer;
  int64_t size;
};
TORCH_MODULE(EncoderLayer);

// Encoder
class EncoderImpl : public torch::nn::Module
{
public:
  EncoderImpl(EncoderLayer layer, int64_t N) : layers(clones(layer, N)), norm(LayerNorm(layer->size))
  {
    for (size_t i = 0; i < layers.size(); ++i) {
      register_module("layer_" + std::to_string(i), layers[i]);
    }
    register_module("norm", norm);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask)
  {
    for (auto& layer : layers) {
      x = layer->forward(x, mask);
    }
    return norm->forward(x);
  }

private:
  std::vector<EncoderLayer> layers;
  LayerNorm norm;
};
TORCH_MODULE(Encoder);

// PositionalEncoding
class PositionalEncodingImpl : public torch::nn::Module
{
public:
  PositionalEncodingImpl(int64_t d_model, double dropout, int64_t max_len = 5000) : dropout(torch::nn::Dropout(dropout))
  {
    // Compute the positional encodings once in log space.
    torch::Tensor pe = torch::zeros({max_len, d_model});
    torch::Tensor position = torch::arange(0, max_len).unsqueeze(1);
    torch::Tensor div_term = torch::exp(torch::arange(0, d_model, 2) * -(std::log(10000.0) / d_model));
    pe.slice(1, 0, d_model, 2) = torch::sin(position * div_term);
    pe.slice(1, 1, d_model, 2) = torch::cos(position * div_term);
    pe = pe.unsqueeze(0);
    register_buffer("pe", pe);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = x + pe.slice(1, 0, x.size(1));
    return dropout->forward(x);
  }

private:
  torch::nn::Dropout dropout;
  torch::Tensor pe;
};
TORCH_MODULE(PositionalEncoding);

// Embedding with scaling
class EmbeddingsImpl : public torch::nn::Module
{
public:
  EmbeddingsImpl(int64_t d_model, int64_t vocab) : lut(torch::nn::Embedding(vocab, d_model)), d_model(d_model) { register_module("lut", lut); }

  torch::Tensor forward(torch::Tensor x) { return lut->forward(x) * std::sqrt(d_model); }

private:
  torch::nn::Embedding lut;
  int64_t d_model;
};
TORCH_MODULE(Embeddings);

// Full Transformer Model
class TransformerBranchPredictorImpl : public torch::nn::Module
{
public:
  TransformerBranchPredictorImpl(int64_t src_vocab, int64_t tgt_vocab, int64_t d_model, int64_t N, int64_t h, int64_t d_ff, double dropout = 0.1)
      : src_embed(Embeddings(d_model, src_vocab)), pos_encoder(PositionalEncoding(d_model, dropout)),
        encoder_layer(EncoderLayer(
            d_model, torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(d_model, h)),
            torch::nn::Sequential(torch::nn::Linear(d_model, d_ff), torch::nn::Functional(torch::relu), torch::nn::Linear(d_ff, d_model)), dropout)),
        encoder(Encoder(encoder_layer, N)), fc_out(torch::nn::Linear(d_model, tgt_vocab))
  {
    register_module("src_embed", src_embed);
    register_module("pos_encoder", pos_encoder);
    register_module("encoder", encoder);
    register_module("fc_out", fc_out);
  }

  torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask)
  {
    torch::Tensor x = src_embed->forward(src);
    x = pos_encoder->forward(x);
    x = encoder->forward(x, src_mask);
    x = x.mean(1); // Global average pooling
    x = fc_out->forward(x);
    return torch::sigmoid(x);
  }

private:
  Embeddings src_embed;
  PositionalEncoding pos_encoder;
  Encoder encoder;
  EncoderLayer encoder_layer;
  torch::nn::Linear fc_out;
};
TORCH_MODULE(TransformerBranchPredictor);

#endif // TRANSFORMER_H
