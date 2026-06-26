import torch

model = torch.load("best.pt")
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(model_quantized, "best_quantized.pt")
