"""B1 XLS-R anti-spoofing model definition matching the Kaggle notebook."""
import torch
import torch.nn as nn
from transformers import AutoModel


class AAAADClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(backbone.config.hidden_size, 2)

    def forward(self, input_values, attention_mask):
        use_mask = self.backbone.config.feat_extract_norm == "layer"
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask if use_mask else None,
        )
        hidden = outputs.last_hidden_state
        frame_mask = self.backbone._get_feature_vector_attention_mask(
            hidden.shape[1], attention_mask
        )
        frame_mask = frame_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * frame_mask).sum(dim=1)
        pooled = pooled / frame_mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


def load_b1(checkpoint_path, model_name="facebook/wav2vec2-xls-r-300m",
            device="cpu"):
    """Load the exact B1 architecture and its saved model_state_dict."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    backbone = AutoModel.from_pretrained(model_name)
    model = AAAADClassifier(backbone)
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint has no 'model_state_dict' entry.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint
