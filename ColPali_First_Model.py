
### Import Packages
from PIL import Image
import torch

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

### Step 1: ColPali Model laden

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


### Step 2: PDF zu PNG Bilder umwandeln

import fitz  # PyMuPDF
from pathlib import Path

# Pfade festlegen
pdf_path = Path("/Users/beatkessler/Desktop/Doravine_MSD") # safed Doc on the Desktop
out_dir = Path("/Users/beatkessler/Desktop/pages")  # Output Folder

out_dir.mkdir(parents=True, exist_ok=True)


## Test ob PDF lädt
import fitz # Toll für PDF handling

pdf_path = "/Users/beatkessler/Desktop/Doravirine_MSD.pdf"

doc = fitz.open(pdf_path)
print("Seitenanzahl:", len(doc))
doc.close()                             # Test bestanden


### Step 3: PDF wird in png gerendert

import fitz
pdf_path = "/Users/beatkessler/Desktop/Doravirine_MSD.pdf"
doc = fitz.open(pdf_path)

page = doc[0]  # erste Seite
mat = fitz.Matrix(2, 2)  # Zoomfaktor 2 = ~144 dpi
pix = page.get_pixmap(matrix=mat)

pix.save("/Users/beatkessler/Desktop/page_001.png")
print("Seite 1 als Bild gespeichert auf Desktop: page_001.png")

doc.close()
