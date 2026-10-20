import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Input : Log-Mel spectrogram (batch, 64, 313) or (batch, 1, 64, 313)
    Output: embedding (batch, embedding_dim) for fusion,
            logit (batch,) for audio-only real/fake prediction
    """

    def __init__(self, embedding_dim=256, dropout=0.3):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel dimension
        x = self.features(x)
        x = self.pool(x)
        emb = self.embedding(x)
        logit = self.classifier(emb).squeeze(1)
        return emb, logit


if __name__ == "__main__":
    model = AudioEncoder()
    dummy = torch.randn(4, 64, 313)
    emb, logit = model(dummy)
    print("Embedding:", tuple(emb.shape))
    print("Logit:", tuple(logit.shape))