# BRATS Reporting

**Paper:** [BTReport: A Framework for Brain Tumor Radiology Report Generation with Clinically Relevant Features](https://arxiv.org/pdf/2602.16006) (arXiv:2602.16006).

## Pipeline

**Inputs (BraTS):** Pre-operative multimodal MRI as NIfTI (T1, T1Gd, T2, T2-FLAIR), co-registered, 1 mm³ isotropic, skull-stripped, with manual tumor labels: necrotic / non-enhancing core (NCR/NET), peritumoral edema (ED), and GD-enhancing tumor (ET). Overall survival and related clinical fields are available per BraTS task definitions.

1. **Deterministic feature extraction** — Quantities such as lesion size, subregion fractions, mass effect, and location are computed from the volumes and segmentations (no vision-language model for “reading” the scan).
2. **Report generation** — A large language model is used only to turn those features into structured wording and narrative (syntax/formatting), not to invent findings from pixels.

**Output:** Natural-language radiology reports (and datasets like BTReport-BraTS that pair BraTS imaging with reports produced this way).

---

## Example Base Features:
- Tumor: Glioma
- Size: 5.9 x 5.2 x 5.0 cm
- Number of Lesions: 1
- Percent NCR/ED/ET: 13/21/65
- Survival Time: 140 days

## Derived Features (paper example)

- Max shift: 10 mm to the left
- Level: Fourth ventricle
- Edema crosses midline: True
- Asymmetric ventricles: True
- Enlarged ventricles: False

## Advanced Features (paper example)

- Location: Right temporal lobe
- Ventricles Invaded: False
- Proportion enhancing: 33%
- Deep WM Invaded: False
- Multifocal: False

## Reasoning (paper excerpt)

> There is a solitary, markedly enhancing lesion centered in the right temporal, parietal, insular and frontal lobes, measuring 5.2 x 5.0 x 5.9 cm (AP x TV x CC)…


## Full Report:
MASS EFFECT & VENTRICLES: There is an approximately 10 mm right-to-left midline shift at the level of the fourth ventricle. The
right lateral ventricle is effaced, producing ventricular asymmetry. No additional ventricular enlargement is noted.
BRAIN/ENHANCEMENT: There is a solitary, markedly enhancing lesion centered in the right temporal, parietal, insular and frontal
lobes, measuring 5.2 x 5.0 x 5.9 cm (AP x TV x CC). The enhancing component is thick (>3 mm) and comprises roughly 21 % of the
lesion, while a central necrotic area accounts for about 13 % of the mass. The lesion demonstrates cortical involvement and deep
white-matter invasion, extending into the right thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens area,
ventral dorsal caudate, and brain-stem. Ependymal (ventricular) invasion of the right lateral ventricle and right inferior lateral
ventricle is present.
A large surrounding vasogenic edema occupies approximately 65 % of the lesion volume and crosses the midline. No enhancing
satellite lesions are identified, and the enhancing portion does not cross the midline.