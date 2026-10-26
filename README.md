# FieldReady — Reliable Tomato Leaf Disease Detection

Deep learning assignment, due **October 24, 2026**. Classifies a tomato leaf photo as **Healthy**, **Early Blight**, or **Late Blight**, with validation-selected abstention. This is a screening aid, not a definitive diagnosis.

## Quick start in VS Code

Use Python 3.11 or 3.12. Run from this repository directory. Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m fieldready.train --include-other --output artifacts/fieldready_rejection
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`. Training automatically uses CUDA when available, otherwise CPU (much slower). For CPU-only Linux installation, install `torch torchvision --index-url https://download.pytorch.org/whl/cpu` with pip before the requirements. Keep torch and torchvision as a compatible pair. Default training is 20 epochs, batch size 64, learning rate 0.0003, patience 5; use `--batch-size 32` for limited memory or `--workers 0` for worker issues. Use a fresh `--output` directory for each run.

Evaluate both datasets with the frozen checkpoint and threshold:

```bash
mkdir -p data
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git data/PlantDoc-Dataset
python -m fieldready.evaluate --model-dir artifacts/fieldready_rejection --plantdoc data/PlantDoc-Dataset --dataset both
python app.py --model-dir artifacts/fieldready_rejection
```

Open the local URL printed by Gradio (normally http://127.0.0.1:7860). No model ships with the project; train first or copy the trained artifact folder from Colab. Inference loads saved weights without downloading ImageNet weights. GPU evaluation: append `--device cuda`. PlantVillage-only evaluation: `python -m fieldready.evaluate --dataset plantvillage`. External-only evaluation: `python -m fieldready.evaluate --dataset plantdoc --plantdoc data/PlantDoc-Dataset`.

## Free Colab GPU workflow

Open **DLproject.ipynb** in Colab and select **Runtime → Change runtime type → T4 GPU** (availability is not guaranteed). Upload a ZIP of this project, or clone your own Git repository. Create a clean upload ZIP locally (Linux/macOS):

```bash
zip -r fieldready-project.zip fieldready tests app.py requirements.txt README.md VERIFICATION.md DLproject.ipynb .gitignore -x '*/__pycache__/*'
```

The notebook contains the same exact commands and a final artifact download cell. If using a ZIP named `fieldready-project.zip` with the project files at its root:

```python
from google.colab import files
files.upload()  # select fieldready-project.zip
!unzip -q fieldready-project.zip -d /content/fieldready
%cd /content/fieldready
!pip -q install -r requirements.txt
!nvidia-smi
!python -m fieldready.train --include-other --output artifacts/fieldready_rejection --batch-size 64 --workers 2
!git clone https://github.com/pratikkayal/PlantDoc-Dataset.git data/PlantDoc-Dataset
!python -m fieldready.evaluate --model-dir artifacts/fieldready_rejection --plantdoc data/PlantDoc-Dataset --dataset both --device cuda
!python app.py --model-dir artifacts/fieldready_rejection --share
```

The app cell stays running; stop it before running subsequent cells. `--share` deliberately creates a public temporary Gradio URL. Local use does not require it. Download results before the Colab runtime expires:

```python
!zip -qr /content/fieldready-results.zip artifacts/fieldready_rejection
files.download('/content/fieldready-results.zip')
```

Extract this ZIP into the local project root, then run `python app.py`. Allow several GB of disk for the upstream archive and Python packages; download and epoch times depend on the available runtime. The lightweight model uses 224×224 images and full fine-tuning; no GPU-specific extensions or mixed precision are required.

## Data protocol and leakage prevention

Source: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage), **color** configuration, revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7`. Modern Hugging Face datasets versions do not execute its legacy loader. Instead, `fieldready/data.py` downloads that repository's `data.zip`, `splits/color_train.txt`, `splits/color_test.txt`, and `leaf_grouping/leaf-map.json` through `huggingface_hub`, reproducing the pinned [upstream leaf-ID logic](https://huggingface.co/datasets/mohanty/PlantVillage/blob/9e97599868962bd0079b8db4b7f1efa9185fa1e7/plant_village.py). The three requested screening classes use these labels:

| Source label | Output / index |
| --- | --- |
| `Tomato___healthy` | Healthy / 0 |
| `Tomato___Early_blight` | Early Blight / 1 |
| `Tomato___Late_blight` | Late Blight / 2 |

The supplied test membership is preserved exactly. With `--include-other`, the seven remaining PlantVillage tomato labels are pooled into index 3, **Other tomato disease**: Bacterial spot, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, and mosaic virus. This is a rejection category, not a fourth diagnosis. This extension supersedes the initial three-class-only training scope in response to Leaf Mold being incorrectly forced into a blight class. Without that flag the original three-class protocol remains available.

The first deterministic five-fold stratified **group** split of official training data yields approximately 80% training / 20% validation; groups are supplied physical `leaf_id` values. Images from one mapped leaf cannot cross partitions. The actual upstream metadata contains some training leaves also present in the supplied test set. Before splitting training, the project explicitly quarantines those training images and records every excluded path in `split_audit.json`, preserving every official test image. Remaining path and leaf-ID intersections across all three partitions cause a hard error. Split membership is saved and evaluation rechecks the official test manifest. Upstream fallback IDs are used when no mapping exists; unknown physical relationships cannot be guaranteed by metadata. Quarantine uses only identities, never test predictions or labels to select the model. All exclusions are reported.

Training uses random resized crops, horizontal flips, mild rotation and color jitter, followed by ImageNet normalization. Validation, test, and inference use identical RGB conversion, EXIF orientation, bilinear short-side resize to 256, center crop to 224, and normalization. Python, NumPy, torch and worker seeds are set; CUDA deterministic convolution settings are enabled. Exact bitwise reproducibility across devices/library versions is not promised.

Model: torchvision ImageNet-pretrained **MobileNetV3-Small**, replacing the final classifier with four logits when `--include-other` is enabled; all parameters are fine-tuned with AdamW and inverse-frequency weighted cross entropy for the four-class model (ordinary cross entropy for the original three-class baseline). The checkpoint with the highest validation macro F1 wins; early stopping also uses only validation macro F1. Test images are not loaded by training.

## External evaluation: original cropped PlantDoc

Use [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset), the original **cropped classification** repository, not the object-detection dataset or a relabeled mirror. Its actual `train` and `test` folder names were inspected. Exact mappings are:

| Actual folder | Output |
| --- | --- |
| `Tomato leaf` | Healthy |
| `Tomato Early blight leaf` | Early Blight |
| `Tomato leaf late blight` | Late Blight |

`Tomato leaf` is the source's unqualified healthy category, not an independent expert verification of each image. Other crops are excluded. The main diagnosis report excludes other diseases; the four-class model also gets a separate supplementary report on the exact `Tomato mold leaf` folder, mapped to the rejection category. Matching is exact and case-sensitive; unexpected missing folders cause an error. Evaluation saves the complete folder inventory, selected image paths, source split names, and checkout commit. Both original folders are pooled into one **external evaluation-only** set: “train” is just its upstream directory name. No PlantDoc image enters training, validation, checkpoint selection, or threshold selection. Preserve a checkout for repeatability; reports record its revision. Corrupt images fail visibly instead of silently reducing the test set.

## Abstention and evaluation

After restoring the best checkpoint, select a threshold from validation maximum softmax scores. Choose the threshold with maximum coverage among those achieving at least **99% empirical accepted accuracy** and accepting at least **30 validation images**. These defaults are declared before testing and can be changed with `--target-accuracy` and `--min-accepted`. An image is accepted only if the winning class is one of the three requested diagnoses and `score >= threshold`. The Other category always abstains, even at a score of 100%. Threshold selection includes all validation classes, so other diseases mistakenly accepted as diagnoses count as errors. Tied scores are accepted together. If no candidate meets the policy, threshold 1.000001 abstains on all images. The validation curve and selection status are saved.

**A softmax score does not guarantee correctness.** The 99% target is an empirical selection rule, not a statistical guarantee or a calibrated probability. Selection on the same validation set can be optimistic; report held-out results honestly. Domain shift, unrelated photos, unknown diseases, mixed symptoms, and poor lighting can cause confident errors. The auxiliary Other category covers only the diseases seen during training. It is not a general tomato disease detector or a validated detector of every unfamiliar input. Re-taking a photo cannot guarantee a correct result; seek qualified advice before treatment.

Each test dataset gets overall accuracy, macro F1, per-class precision/recall/F1/support, a raw confusion matrix (rows=true, columns=predicted), and a PNG. Overall metrics include every image regardless of abstention. Selective metrics use the **same frozen validation threshold**: coverage = accepted / total; accepted accuracy = correct accepted / accepted. Zero accepted predictions yield `null` accuracy, not zero or a fabricated value. Compare PlantVillage and PlantDoc to discuss laboratory-to-field domain shift. Do not tune after seeing PlantDoc scores and still describe it as untouched evaluation.

## Files and submission

- `fieldready/train.py`: practical CLI training and model/threshold selection.
- `fieldready/data.py`, `model.py`, `runtime.py`: splits, transforms, model and data loading.
- `fieldready/evaluate.py`, `metrics.py`: held-out and external evaluation.
- `fieldready/inference.py`: reusable `Predictor(...).predict(PIL_image)`.
- `app.py`: Gradio upload, decision/abstention, and all model class scores.
- `DLproject.ipynb`: Colab execution guide.
- `tests/`: leakage, threshold, and runtime checks.

A completed run produces `best.pt`, `metadata.json` (class order, preprocessing, threshold, provenance), `splits.json`, `history.json`, `threshold_selection.json`, and `reports/` containing metrics, confusion matrices, and raw prediction arrays. Datasets, cached files, and checkpoints are ignored by Git. Submit code and notebook plus generated reports, discussion and screenshots; supply model artifacts separately if your instructor requires a runnable demo. Cite both sources and respect their dataset licenses (PlantVillage card lists CC BY-SA 3.0; PlantDoc includes its own LICENSE.txt).

No disease-classification performance metrics are claimed before real training and evaluation. See `VERIFICATION.md` for checks performed in the build environment and remaining validation steps.

## Troubleshooting incorrect predictions

The original three-class 95% validation-accuracy policy could select threshold zero when the entire validation set passed that target. This disabled abstention. The updated selector uses observed validation confidences only and the default target is 99%. Neither change guarantees correctness on real photos: even near-100% softmax scores can be wrong.

To update an existing trained model's policy without retraining, using its saved validation partition and local PlantVillage archive:

```bash
python -m fieldready.calibrate --target-accuracy 0.99
python -m fieldready.evaluate --dataset plantvillage
python app.py
```

Stop the previous app process and restart it to load the new metadata. Calibration preserves the previous settings and reports under `policy_backups/`, keeps model weights unchanged, and reads validation images only. Rerun evaluation after policy changes; old reports contain the old threshold. Do not use test or PlantDoc results to choose a threshold. A wrong real-world prediction needs the original image and expected class for investigation; non-tomato images and diseases absent from training remain outside the learned scope.

## Retraining after the Leaf Mold failure

The current run is saved separately at `artifacts/fieldready_rejection`; the original three-class model remains at `artifacts/fieldready`. Retraining starts from ImageNet weights rather than the previous fitted model, so changing the grouped validation split cannot leak earlier training into the new validation set. All original test images remain held out. The pooled Other category has inverse-frequency loss weighting so its larger sample count does not dominate the three screening classes.

Reproduce locally (CPU settings used for this run):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=data/torch python -m fieldready.train --include-other --output artifacts/fieldready_rejection_new --workers 0 --epochs 20
python -m fieldready.evaluate --model-dir artifacts/fieldready_rejection_new --dataset both --plantdoc data/PlantDoc-Dataset --workers 0
python app.py --model-dir artifacts/fieldready_rejection_new
```

For a four-class checkpoint, evaluation writes the original three-diagnosis test report, a separate all-tomato report, and a Leaf Mold-only rejection report. Confusion matrices include the fourth prediction column, so rejection-class predictions are never silently dropped. The headline macro F1 averages the classes actually present in that evaluation set; `macro_f1_classes` records them. Selective accuracy counts an accepted other-disease image as incorrect. The Leaf Mold rejection rate measures how often its images receive Uncertain, whether from the Other class or the confidence threshold. The main PlantDoc report uses the original three exact matching folders. A separate supplementary rejection report uses `Tomato mold leaf`, mapped to Other (it is not one of the supported diagnoses). Both remain evaluation-only.

Adding this rejection category cannot guarantee that every Leaf Mold photo is rejected. The screenshot is not an independently verified source image and has not been used as a training sample. Check measured held-out rejection rates rather than promising a fix for every photo.
