# Tree-Detection-Samgeo
Repository for counting trees in Lüneburg using geodata with the samgeo library 

Dieses Projekt nutzt State-of-the-Art KI-Modelle (**Grounding DINO** & **Segment Anything - SAM**), um Bäume in Luftbildern automatisch zu detektieren und zu segmentieren. Das Setup ist für die Ausführung auf GPU-Servern mittels Docker optimiert.

## 📁 Projektstruktur

- `gpu_app/`: Hauptanwendung mit Backend und Frontend.
  - `src/prototype/`: Streamlit-App und Pipeline-Logik.
  - `src/demo/`: Testskripte für DINO und SAM.
- `storage/`: (Extern gemountet) Speicherort für Bilddaten und Modell-Checkpoints.

---

## 🚀 Deployment mit Docker

Das Projekt ist so konfiguriert, dass es in einem isolierten Container mit GPU-Unterstützung läuft.

### Voraussetzungen
- NVIDIA GPU & [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- Docker & Docker Compose

### Starten der Anwendung
1. **Repository klonen:**
   ```bash
   git clone [https://github.com/BentMildner/AITreeCounting.git](https://github.com/BentMildner/AITreeCounting.git)
   cd AITreeCounting