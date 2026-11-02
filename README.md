# Can a pretrained network beat a golden template? Anomaly detection on MVTec AD

Poster summary: [link to be added]

I spent 18 years around inspection equipment, where "find the part that does not look like the good ones" is solved with golden templates, filters and a lot of tuning. I wanted to see how far a pretrained neural network gets on the same task with no defect examples at all, only good images. On six MVTec AD categories, a nearest-neighbor method on pretrained features beat my golden-template pipeline clearly on four of them (a tuned classical method came close on two of those), and PatchCore was the best overall, but it takes about fifty times longer to fit than the nearest-neighbor method (466 s against 9 s on grid, CPU). What surprised me: the biggest single gain (grid, AUROC 0.77 to 1.00) came from understanding one artifact at the image border, not from a better model. The fix has a price: a defect lying only in the ignored border cannot be found (one of the 57 defective grid images). Built with Claude as a coding assistant; the notebook is the full record.

## Just read it

Both studies are available as documents that open in any browser and need nothing installed:

- [`mvtec_ad_comparison_report.html`](mvtec_ad_comparison_report.html), the first study: text, tables and all figures, with only the three most important code cells shown. The write-up is the last section.
- [`mvtec_ad_robustness_report.html`](mvtec_ad_robustness_report.html), the second study (see below): text, tables and all figures, no code. The write-up is the last section.

## Second study: what happens when the images change

[`mvtec_ad_robustness.ipynb`](mvtec_ad_robustness.ipynb) asks what survives when the light drifts, a lens gets dirty or a part sits off. The test images are changed in 14 simulated ways (lighting, contrast, colour, blur, noise, dirt on the lens, dust, shift, rotation), three strengths each, and scored with models and thresholds that were fixed beforehand, for the nearest-neighbour method and the golden template. In short: the ranking of good and defective parts survives most changes much better than the threshold does; the template breaks on any change of brightness and the neural method on blur, dirt on the lens and a rotated part; normalising the brightness made the template the most stable method tried, at the price of finding fewer defects; another pretrained network (DINOv2) ignores lighting but reads dirt on the lens and dust as defects; and a handful of reject samples can retune a threshold, at the price of rejecting more good parts. The write-up is the last section of the notebook (also as [`mvtec_ad_robustness_report.html`](mvtec_ad_robustness_report.html)). The notebook is committed without outputs; the stored results (`robustness_batch*.csv`, `robustness_scores*/`) let it run from the top in about 15 minutes.

**Follow-up.** [`robustness_followup.ipynb`](robustness_followup.ipynb) asks two more questions about the nearest-neighbour method, on three objects and five seeds (the seed changes the split of the good training images and the random changes). *Test A:* a brightness normalisation in front of the feature extraction (a gain and offset fitted per image, or also a linear shading) lowers the share of good parts rejected over the 14 changes from 0.26 to 0.18 and raises the AUROC from 0.84 to 0.86; the effect is clear for brighter lighting (0.46 to 0.09) and doubled contrast (0.87 to 0.21), but a shift gets worse (0.56 to 0.74, and 0.97 with the shading term), and the share of defects caught once the threshold is set again does not change beyond the seed spread. *Test B:* refitting the memory with changed copies of the good training images lowers the share of good parts rejected at the old threshold for blur (0.33 to 0.03 with a mixed memory), brighter lighting and dirt on the lens, at the price of about 0.02 AUROC on unchanged images and 2 to 10 times the fitting time; it was not tested on changes that were not in the memory. The notebook states for every result whether it is larger than the seed spread.

## Run it

1. Download MVTec AD (manual download from the [dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad), the licence is accepted there). Only six categories are used: transistor, pill, metal_nut, leather, carpet, grid.
2. Point the environment variable `MVTEC_AD_ROOT` at the folder that holds the category folders (or edit `PATH_ROOT` in the first code cell).
3. `pip install -r requirements.txt` (Python 3.14, tested on Windows, CPU only).
4. Open `mvtec_ad_comparison.ipynb` and run it from the top. The PatchCore fits are slow (about an hour for the table and about an hour more for the per-image scores), so their results are cached in `patchcore_results.csv` and `patchcore_scores/`; delete them to repeat the fits.

## Files

| | |
|---|---|
| `mvtec_ad_comparison.ipynb` | the notebook, committed without outputs |
| `mvtec_ad_comparison_report.html`, `mvtec_ad_robustness_report.html` | readable versions of the two notebooks, with figures (see above) |
| `mvtec_ad_robustness.ipynb` | second notebook: how the nearest-neighbour method copes with simulated changes of lighting, colour, camera, lens and part including the comparison with the golden template and shifted and rotated parts |
| `helpers.py`, `robustness_viz.py`, `threshold_flow.py` | code shared by the second notebook: the methods and the simulated variations, the measurements and plots, the threshold-with-reject-samples simulation |
| `robustness_batch.csv`, `robustness_scores/` | results of the batch variations for the neural method: summary per condition and category, and the score of every image |
| `robustness_batch_template.csv`, `robustness_scores_template/` | the same for the golden-template method |
| `robustness_batch_geometry*.csv`, `robustness_scores_geometry*/` | the same for shifted and rotated parts, for both methods |
| `robustness_batch_template_photo.csv`, `robustness_batch_neural_aligned.csv`, `robustness_scores_*/` | the two candidates for a more robust method (template with brightness normalisation, neural method on aligned images), three objects |
| `robustness_batch_convnext.csv`, `robustness_batch_dinov2.csv` | the neural method with the features of two other pretrained networks (ConvNeXt-tiny, DINOv2 small), three objects |
| `robustness_followup.ipynb` | follow-up with five seeds: a brightness normalisation before the feature extraction (Test A) and refitting the memory with changed good images (Test B), nearest-neighbour method, three objects |
| `helpers_followup.py`, `robustness_followup_run.py` | code for the follow-up: the normalisation and the refit memories; the runner (several resumable worker processes, `merge` writes the CSV files) |
| `robustness_batch_nn_norm.csv`, `robustness_batch_nn_refit.csv`, `robustness_batch_nn_*_scores/` | results of Test A and Test B: summary per condition, category and seed, and the score of every image |
| `shift_check.py`, `shift_frame_check.csv` | check that the damage of a simulated shift or rotation is mostly the mirrored strip it uncovers |
| `requirements.txt` | package versions the notebook was run with |
| `patchcore_results.csv`, `patchcore_scores/` | cached PatchCore results |
| `classical_ablation.csv` | table of the classical steps, written by the notebook |
| `tools/strip_outputs.py` | git filter that strips notebook outputs when committing |
| `LICENSE` | MIT licence for the code |

If you work on a clone, set the filter once: `git config filter.nbclean.clean "python tools/strip_outputs.py"` and `git config filter.nbclean.required true`.

## Data and licences

The MVTec AD dataset is licensed CC BY-NC-SA 4.0 (attribution, non-commercial use, share-alike) and must be cited: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and Pattern Recognition, 2019. The dataset itself is not part of this repository. The figures in the reports show MVTec AD images, so the same licence applies to them.

The code is licensed under the MIT licence, see [`LICENSE`](LICENSE). The reports and their figures follow the dataset's licence (CC BY-NC-SA 4.0), not the MIT licence.
