
### Import Packages

from PIL import Image
import torch

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

### Modell

model_name = "vidore/colpali-v1.2"


# Model läuft auf CPU
device = "cpu"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # CPU arbeitet mit float32 stabil
    trust_remote_code=True,
).to(device).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

print("run")
