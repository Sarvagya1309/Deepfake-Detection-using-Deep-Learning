## Datasets Used

- FaceForensics++
- Celeb-DF
- Deepfake Detection(DFD)

⚠️ Datasets are not included due to size and licensing restrictions.

### Expected structure
data/
 ├── real/
 └── fake/

Each directory should contain either raw videos or extracted face frames,
depending on the preprocessing stage.

Face frames are extracted using SCRFD (InsightFace) and resized to 224×224
prior to feature extraction.

