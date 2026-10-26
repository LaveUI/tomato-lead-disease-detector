# Build verification — October 17, 2026

## Verified

- Installed dependencies into the workspace `.venv`; CPU torch and torchvision import and run.
- Seven unittest checks passed: reproducible grouped splitting, overlap rejection, training-only quarantine, upstream leaf-ID aliases, threshold optimization, tied-score/no-feasible-policy behavior, and a synthetic integration test.
- Integration test ran a full one-epoch CPU training loop using generated images and randomly initialized MobileNetV3, selected/saved/reloaded the checkpoint and threshold, ran the inference API, generated external-evaluation JSON/NPZ/confusion PNG files, and built the actual Gradio interface with server launch mocked. Synthetic scores are not assignment performance results; temporary artifacts were removed by the test.
- Downloaded actual ImageNet MobileNetV3-Small pretrained weights (about 10 MB) and confirmed a CPU forward pass with output shape `[1, 3]`.
- Downloaded the pinned real PlantVillage color manifests and leaf map. Selected 3,580 supplied training images and 920 supplied test images. Found five training images with leaf IDs also present in test. Explicit training-only quarantine leaves 3,575 eligible images; seed 42 yields 2,863 train and 712 validation images. No remaining path/leaf-ID intersections exist across the three partitions.
- 1,577 of the 4,495 retained images use upstream fallback identifiers; the metadata cannot establish all physical-leaf relationships. Group isolation is guaranteed for the identifiers available, not unknown relationships.
- Inspected the real remote archive central directory through HTTP range requests: all 4,500 selected original manifest paths exist in the 2,184,723,441-byte ZIP. No archive-prefix assumptions remain untested.
- Inspected actual `train` and `test` folders of the original PlantDoc repository via GitHub API; all three exact selected names exist in both splits.
- CLI help/import checks and Python compilation passed.

## Not verified / still required

- CUDA is unavailable (`torch.cuda.is_available()` returned false). No full real-data training run, trained disease classifier, validation-selected real threshold, or measured disease-classification metrics are supplied.
- The complete 2.18 GB PlantVillage archive and PlantDoc image checkout were not downloaded. Archive paths were checked, but complete download through `huggingface_hub`, decoding every real image, dataset evaluation, and Colab GPU execution still need the documented run.
- Gradio interface construction and prediction callback dependencies were tested; a live browser upload and public Colab share link were not tested.
- Network initially failed inside the restricted shell; approved network access allowed metadata, packages, archive-directory inspection, and pretrained-weight verification. This is not an offline-ready dataset installation.
- The build host uses Python 3.14, torch 2.14.1+cpu, torchvision 0.29.1+cpu, scikit-learn 1.7.2, and Gradio 6.30.0. The README recommends Python 3.11/3.12 for broad runtime availability. Package ranges are not a fully locked environment; each training artifact records installed library versions.

Before October 24: run the Colab notebook, inspect both datasets' generated metrics and confusion matrices, discuss the field-domain gap and abstention coverage, download the artifact ZIP, and test a real local image upload. Keep test results out of further model/threshold selection. Include actual results and the metadata limitations in the submission.

## Follow-up — October 18, 2026

A trained checkpoint and completed 17-epoch training history are now present (best epoch 12). The real local PlantVillage archive is also present. This supersedes the original build-time statements about missing training and archive artifacts.

Investigated reported incorrect predictions. Saved threshold 0 accepted every photo. Updated selection to consider observed validation confidences instead of a zero candidate, and raised the declared default validation accepted-accuracy target from 95% to 99%. Recomputed predictions on the saved 712-image validation partition before examining test performance. New threshold: 0.6931474804878235; validation accepted 709/712, with 99.0127% accepted accuracy. Prior policy and reports were backed up under the artifact folder's `policy_backups/`; trained weights were not changed.

Then evaluated the unchanged official 920-image test partition: accuracy 99.1304%, macro F1 0.990325; accepted 913/920 (99.2391% coverage), accepted accuracy 99.4524%. Outputs are in `artifacts/fieldready/reports/plantvillage_test.*`. These are PlantVillage results, not real-world reliability estimates. Some validation mistakes had very high confidence, so this policy cannot eliminate confident errors.

Eight tests pass, including a regression test that easy validation data no longer selects threshold zero. PlantDoc evaluation and inspection of the user's failing photo remain outstanding. Restart the app to load the changed policy.
