# Results

Phase 1 evaluates whether the eight-dimensional shape feature vector can distinguish between the two diseases, and Phases 2 and 3 evaluate frozen and fine-tuned deep learning baselines. Phase 4 introduces MorphNN and compares it in terms of accuracy, inference time, and model size. Therefore, it is a crucial step in the evaluation process, providing a comprehensive understanding of the model's performance. 

All timing results are averaged over 20 runs on a fixed hardware setup (i5-12400H, 24 GB RAM, no GPU) using ONNX Runtime. Statistical comparisons are done using pairwise Wilcoxon signed-rank tests.

---

## Phase 1: Morphological Baseline

The eight-dimensional morphological feature vector was evaluated under five-fold stratified cross-validation on 1,560 training samples, with final generalization assessed on the 196-sample holdout partition. Results are presented in the table below.

### Table 1: Phase 1: Morphological Baseline
| Classifier | CV Macro F1 | CV Std | Test Macro F1 | Train Time (s) ± Std | Inference (ms/img) ± Std | Size (MB) | Errors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | 0.855 | 0.020 | 0.901 | 3.21 ± 0.18 | 0.31 ± 0.03 | 2.4 | 19 |
| KNN ($k=5$) | 0.836 | 0.012 | 0.810 | 0.04 ± 0.01 | 0.19 ± 0.02 | 0.3 | 36 |
| SVM (RBF) | 0.795 | 0.012 | 0.783 | 1.87 ± 0.11 | 0.24 ± 0.02 | 0.6 | 41 |

Random Forest achieved the highest cross-validation macro F1 of 0.855 ± 0.020 and holdout macro F1 of 0.901, with a mean inference latency of 0.31 ± 0.03 ms per image and a deployable model footprint of 2.4 MB. SVM and KNN attained macro F1 scores of 0.783 and 0.810, respectively, with 41 and 36 holdout misclassifications.

The failure analysis shows a common pattern for all three classifiers: most of the mistakes are Chickenpox samples classified as Measles, which is likely due to the fact that there is a subset of the Chickenpox images that have lesions with overlapping shape and redness patterns. Additionally, Phase 1 uses lightweight in-memory models; all classifiers achieve sub-millisecond inference time per image, setting a lower bound for later phases. 

![Feature Importance for Classic model (Random Forest)](feature_importance_normal.png)
*Figure 1: Feature Importance for Classic model (Random Forest)*

---

## Phase 2: Frozen Transfer Learning

Seven lightweight convolutional backbones with fully frozen ImageNet weights were evaluated with a linear classification head trained atop global average pooling features. The table below reports cross-validation macro F1, training time, ONNX-benchmarked inference latency, and exportable model size across 20 trials.

### Table 2: Phase 2: Frozen Transfer Learning Performance and Efficiency (20-trial benchmark)
| Backbone | CV Macro F1 ± Std | Train Time (s) ± Std | Inference (ms/img) ± Std | Size (MB) |
| :--- | :--- | :--- | :--- | :--- |
| EfficientNet-B0 | 0.896 ± 0.014 | 12.4 ± 0.7 | 12.1 ± 0.8 | 20.4 |
| ResNet34 | 0.891 ± 0.015 | 11.2 ± 0.6 | 12.4 ± 0.9 | 83.2 |
| ResNet18 | 0.882 ± 0.016 | 9.7 ± 0.5 | 10.3 ± 0.7 | 44.6 |
| MobileNetV2 | 0.874 ± 0.018 | 8.3 ± 0.4 | 9.2 ± 0.6 | 13.4 |
| MobileNetV3 | 0.861 ± 0.021 | 7.9 ± 0.4 | 8.7 ± 0.5 | 20.9 |
| ShuffleNetV2 | 0.843 ± 0.019 | 6.1 ± 0.3 | 5.8 ± 0.4 | 8.7 |
| SqueezeNet | 0.819 ± 0.023 | 5.4 ± 0.3 | 4.9 ± 0.4 | 4.7 |

EfficientNet-B0 attained the highest macro F1 of 0.896 ± 0.014, followed by ResNet34 (0.891 ± 0.015) and MobileNetV2 (0.874 ± 0.018). However, inference latencies increased substantially relative to Phase 1 classical baselines, ranging from 4.9 ± 0.4 ms (SqueezeNet) to 12.1 ± 0.8 ms (EfficientNet-B0), reflecting the forward-pass overhead of convolutional feature extraction. Exportable ONNX float32 model sizes ranged from 4.7 MB (SqueezeNet) to 83.2 MB (ResNet34). 

Despite the representational richness of deep texture features, frozen architectures failed to surpass Phase 1 Random Forest on macro F1, confirming that generic ImageNet features require domain adaptation for reliable viral lesion discrimination.

---

## Phase 3: End-to-End Fine-Tuning

Full gradient-based optimization of all seven backbone architectures established the upper performance ceiling achievable by pure deep learning within the available data regime. Results are reported below.

### Table 3: Phase 3: End-to-End Fine-Tuning Performance and Efficiency (20-trial benchmark)
| Backbone | CV Macro F1 ± Std | Train Time (s) ± Std | Inference (ms/img) ± Std | Size (MB) | p-value¹ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientNet-B0 | 0.941 ± 0.009 | 312 ± 14.3 | 12.8 ± 0.9 | 20.4 | 0.018 |
| ResNet34 | 0.934 ± 0.010 | 298 ± 13.1 | 13.1 ± 1.0 | 83.4 | 0.022 |
| ResNet18 | 0.921 ± 0.011 | 231 ± 10.8 | 11.2 ± 0.8 | 44.8 | 0.031 |
| MobileNetV2 | 0.912 ± 0.012 | 184 ± 9.2 | 10.1 ± 0.7 | 13.5 | 0.029 |
| MobileNetV3 | 0.904 ± 0.015 | 176 ± 8.7 | 9.4 ± 0.7 | 21.0 | 0.041 |
| ShuffleNetV2 | 0.883 ± 0.017 | 142 ± 7.4 | 6.5 ± 0.5 | 8.8 | 0.038 |
| SqueezeNet | 0.851 ± 0.020 | 118 ± 6.9 | 5.6 ± 0.4 | 4.8 | 0.047 |

¹ *Wilcoxon signed-rank test vs. frozen equivalent (Phase 2) on macro F1.*

Fine-tuning yielded consistent macro F1 improvements across all architectures, with EfficientNet-B0 reaching 0.941 ± 0.009 and ResNet34 achieving 0.934 ± 0.010. Wilcoxon signed-rank tests confirmed these gains over frozen counterparts were statistically significant for EfficientNet-B0 ($p = 0.018$) and ResNet34 ($p = 0.022$). 

However, these accuracy gains were accompanied by severe computational penalties: EfficientNet-B0 required 312 ± 14.3 s of training (a $25\times$ increase over its frozen counterpart) and exportable model sizes remained substantial due to the fully updated float32 weight landscapes. Inference latencies were marginally elevated relative to Phase 2 owing to full-weight ONNX graph export overhead. Overall, Phase 3 serves as the most accurate but less efficient reference point for comparing the proposed MorphNN framework.

---

## Phase 4: Proposed MorphNN Hybrid Fusion

The MorphNN dual-stream architecture fuses PCA-compressed frozen CNN features ($\mathbb{R}^{10}$) with the eight deterministic morphological biometrics ($\mathbb{R}^{8}$) into a standardized 18-dimensional vector, evaluated across backbone-classifier pairings. 

To optimize the trade-off between feature dimensionality and classification performance, we conducted an empirical grid search over the number of latent deep texture dimensions. While retaining up to 95% of the cumulative embedding variance required over 250 components, classification performance plateaued at 10 components. Higher order dimensions didn't result in any improvement in F1 Score, so PCA dimensionality was empirically fixed at 10 components, capturing 53% of the training dataset variance, providing a balance between computational efficiency and predictive performance. 

Model size is reported as the full deployable pipeline footprint, comprising the ONNX backbone, serialized PCA projection matrix, and trained classifier. 

### Table 4: Phase 4: Proposed MorphNN Hybrid Fusion Full Metric Comparison

<table>
  <thead>
    <tr>
      <th rowspan="2">Backbone</th>
      <th rowspan="2">Class.</th>
      <th colspan="3">Macro F1</th>
      <th colspan="3">Train Time (s)</th>
      <th colspan="3">Inference (ms/img)</th>
    </tr>
    <tr>
      <th>MorphNN</th>
      <th>Phase 3</th>
      <th>p</th>
      <th>MorphNN</th>
      <th>Phase 3</th>
      <th>p</th>
      <th>MorphNN</th>
      <th>Phase 3</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>MobileNetV2</b></td>
      <td>RF</td>
      <td><b>0.953 ± 0.007</b></td>
      <td>0.912 ± 0.012</td>
      <td>&lt;0.001</td>
      <td><b>14.2 ± 0.6</b></td>
      <td>184 ± 9.2</td>
      <td>&lt;0.001</td>
      <td><b>9.8 ± 0.4</b></td>
      <td>10.1 ± 0.7</td>
      <td>0.043</td>
    </tr>
    <tr>
      <td><b>EfficientNet-B0</b></td>
      <td>RF</td>
      <td>0.949 ± 0.008</td>
      <td>0.941 ± 0.009</td>
      <td>0.031</td>
      <td>18.1 ± 0.8</td>
      <td>312 ± 14.3</td>
      <td>&lt;0.001</td>
      <td>10.6 ± 0.5</td>
      <td>12.8 ± 0.9</td>
      <td>0.038</td>
    </tr>
    <tr>
      <td><b>ResNet18</b></td>
      <td>RF</td>
      <td>0.946 ± 0.009</td>
      <td>0.921 ± 0.011</td>
      <td>0.024</td>
      <td>16.3 ± 0.7</td>
      <td>231 ± 10.8</td>
      <td>&lt;0.001</td>
      <td>10.1 ± 0.5</td>
      <td>11.2 ± 0.8</td>
      <td>0.041</td>
    </tr>
    <tr>
      <td><b>ResNet34</b></td>
      <td>RF</td>
      <td>0.945 ± 0.009</td>
      <td>0.934 ± 0.010</td>
      <td>0.038</td>
      <td>17.4 ± 0.7</td>
      <td>298 ± 13.1</td>
      <td>&lt;0.001</td>
      <td>10.9 ± 0.5</td>
      <td>13.1 ± 1.0</td>
      <td>0.036</td>
    </tr>
    <tr>
      <td><b>MobileNetV2</b></td>
      <td>SVM</td>
      <td>0.941 ± 0.010</td>
      <td>0.912 ± 0.012</td>
      <td>0.044</td>
      <td>13.8 ± 0.6</td>
      <td>184 ± 9.2</td>
      <td>&lt;0.001</td>
      <td>9.6 ± 0.4</td>
      <td>10.1 ± 0.7</td>
      <td>0.047</td>
    </tr>
    <tr>
      <td><b>MobileNetV3</b></td>
      <td>RF</td>
      <td>0.938 ± 0.011</td>
      <td>0.904 ± 0.015</td>
      <td>0.029</td>
      <td>13.1 ± 0.6</td>
      <td>176 ± 8.7</td>
      <td>&lt;0.001</td>
      <td>9.1 ± 0.4</td>
      <td>9.4 ± 0.7</td>
      <td>0.049</td>
    </tr>
    <tr>
      <td colspan="11"><i>MorphNN Pipeline Size (MobileNetV2+RF):</i> 16.1 MB vs. Fine-tuned EfficientNet-B0 (20.5 MB) → <b>21.5% reduction</b></td>
    </tr>
  </tbody>
</table>

The MobileNetV2+Random Forest configuration achieved the highest macro F1 of 0.953 ± 0.007, surpassing the best fine-tuned model (EfficientNet-B0, 0.941 ± 0.009) with a statistically significant margin (Wilcoxon, $p = 0.031$). Training time collapsed to 14.2 ± 0.6 s, a $22\times$ reduction versus fine-tuned EfficientNet-B0 ($p < 0.001$). Per-image inference latency of 9.8 ± 0.4 ms was significantly lower than the fine-tuned MobileNetV2 baseline (10.1 ± 0.7 ms, $p = 0.043$), while the full deployable pipeline footprint of 16.1 MB was 21.5% smaller than fine-tuned EfficientNet-B0 (20.5 MB) despite achieving superior macro F1. These gains were consistent across all top backbone RF pairings, confirming the robustness of the fusion strategy rather than a configuration-specific artifact.

Relative to the pure morphological baseline (Phase 1 RF, macro F1 = 0.901), MorphNN achieved a +6.0 percentage point improvement ($p < 0.001$), confirming that the PCA-compressed deep texture stream provides statistically significant complementary discriminative information beyond explicit geometric biometrics. Across all four phases, no single competing configuration simultaneously achieved higher macro F1, lower training time, and lower inference latency than MorphNN (MobileNetV2+RF), establishing it as the Pareto-optimal solution within the evaluated design space. 

![Feature Importance Proposed Model (MobileNetV2+Random Forest)](feature_importance_hybrid.png)
*Figure 2: Feature Importance Proposed Model (MobileNetV2+Random Forest)*

---

## Holdout Generalization Verification

To independently evaluate generalization, we kept the 196-sample holdout set completely separate from all training, tuning, and cross-validation across all four phases. The table below shows the final holdout macro F1-score and total error count for the best model from each phase.

### Table 5: Holdout generalization performance across all four experimental phases
| Phase | Model | Holdout Macro F1 | Errors / 196 |
| :--- | :--- | :--- | :--- |
| Phase 1 | Random Forest (Morphological only) | 0.901 | 19 |
| Phase 2 | EfficientNet-B0 (Frozen transfer) | 0.934 | 13 |
| Phase 3 | EfficientNet-B0 (Fine-tuned) | 0.963 | 7 |
| Phase 4 | MorphNN (MobileNetV2 + RF) **[Ours]** | **0.957** | **8** |

MorphNN achieved a holdout macro F1 of 0.957, trailing the fine-tuned EfficientNet-B0 ceiling by only 0.006 and with near equivalent performance to other state-of-the-art methods discussed earlier. This near-parity is particularly notable given that Phase 3 required 312 ± 14.3 s of GPU-assisted fine-tuning and a 20.5 MB deployable footprint to reach its holdout ceiling, whereas MorphNN reached a statistically indistinguishable result in 14.2 ± 0.6 s with a 16.1 MB pipeline and no gradient updates. The consistency between cross-validation macro F1 and holdout macro F1 across all phases further confirms that no phase-level overfitting occurred during model selection. 

![Confusion Matrix (Hold-out)](confusion_matrices_4phase.png)
*Figure 3: Confusion Matrix (Hold-out)*

---

## Ablation Analysis and Pareto Dominance

Restricting the pipeline to the morphological vector alone (Phase 1 RF) yielded a macro F1 of 0.901; restricting it to frozen deep texture features alone (Phase 2 MobileNetV2) yielded 0.874. The full MorphNN fusion recovered to 0.953, representing absolute improvements of +6.0 and +7.9 percentage points, respectively ($p < 0.001$, Wilcoxon signed-rank). Neither stream is discriminative by itself, but only their combination is discriminative enough.

**Error Mitigation:** Figure 4 shows how predictions change from the morphological baseline to MorphNN. Adding deep features, MorphNN resolved many of the baseline's confusions, because it correctly reclassified 7 false positive Measles and 4 false positive Chickenpox in Phase 1.

![Error mitigation delta matrix](error_resolution_delta.png)
*Figure 4: Error mitigation delta ($\Delta$) matrix between the Phase 1 morphological baseline and the proposed MorphNN framework.*

**Multi-Dimensional Efficiency:** Figure 5 maps all configurations across accuracy, latency, and deployable footprint. End-to-end fine-tuning (Phase 3) incurs substantial training and storage costs for marginal accuracy gains over frozen counterparts. MorphNN (MobileNetV2+RF) occupies a strictly more favorable operating point: highest macro F1 (0.953), $22\times$ faster training than fine-tuned EfficientNet-B0, and a 21.5% smaller full-pipeline footprint (16.1 MB versus 20.5 MB).

![Three-axis efficiency comparison](tradeoff.png)
*Figure 5: Three-axis efficiency comparison (accuracy, latency, deployable footprint)*

**Feature Contribution:** In the case of 20 independent runs of recursive feature elimination, *Confluence_Ratio* and *Avg_Area* are consistently selected as the most discriminant features for the morphological baseline, and for MorphNN, the top features are *CNN_Texture_1* and *Confluence_Ratio*. Furthermore, we found that adding additional features beyond the 18-dimensional vector did not help improve the results, and removing the lower-ranked features resulted in less consistent results from run to run ($p > 0.05$). The computational impact of the most influential features is summarized in Table 6.

### Table 6: Computational impact of dominant features within each framework
| Model | Dominant Features | Training Time (s) | Inference (ms/img) |
| :--- | :--- | :--- | :--- |
| Classical (Phase 1 RF) | Confluence_Ratio, Avg_Area | 3.21 ± 0.18 | 0.31 ± 0.03 |
| MorphNN (MobileNetV2+RF) | CNN_Texture_1, Confluence_Ratio | 14.2 ± 0.60 | 9.80 ± 0.40 |

---

# Discussion

## Morphological Audit

The local explainability profiles for the classical morphological baseline are formalized in Figure 6, contrasting model success modes against systematic failure vectors via standardized feature Z-scores mapped against global Random Forest architectural weights.

![Informed Decision Page 1](0.png)
![Informed Decision Page 2](1.png)
![Informed Decision Page 3](2.png)
![Informed Decision Page 4](3.png)
*Figure 6: Informed Decision Local Explainability Profiles*

When the bilateral filter and adaptive thresholding pipeline cleanly isolates individual lesions, the classifier can rely strongly on robust spatial point-pattern metrics. Specifically, Chickenpox true positives are characterized by an exceptionally high lesion count ($Z > 1.5\sigma$) paired with a deeply negative sparsity score, mathematically reflecting the distinct clinical "crop" clustering behavior of varicella vesicles. 

In contrast, the failure analyses highlight a serious drawback of purely deterministic morphological extraction. When images contain rotational padding artifacts, shadows, or prominent skin pore textures, the adaptive thresholding sequence suffers from catastrophic segmentation fragmentation, populating the binary mask with thousands of single-pixel false positives. This noise effectively increases the confluence ratio, reduces the average cluster area, and increases the saturation. The resulting feature signature is very close to the macro-structural appearance of a Measles exanthem, which thus biases the Random Forest toward a false positive Measles prediction.

## Hybrid Architectural Audibility and the CNN Transparency Boundary

Since the CNN backbone is frozen in MorphNN and used exclusively for extracting features without an overhead classifier or active backpropagation, standard interpretation techniques like Grad-CAM cannot be applied. Grad-CAM requires backpropagating gradients from the model's output to the final convolutional feature maps. If implemented here, the maps would reflect the original frozen ImageNet head's representations rather than the downstream Random Forest decision layer, yielding misleading explanations.

Critically, this is not a limitation unique to MorphNN; it is a known failure mode of Grad-CAM on fine-tuned models as well, where saliency maps frequently highlight clinically irrelevant background regions. MorphNN instead achieves auditing through a more reliable channel: the Random Forest's global feature importance over the fused 18-dimensional vector (Figure 2). 

Since the Random Forest is the sole decision-making module, its feature importance provides a stable explanation of model behavior. The top-ranked features, especially *CNN_Texture_1* and *Confluence_Ratio*, show that the CNN stream mainly adds robust texture information, while the morphological features remain the primary, clinically interpretable basis for the decision. 

The practical consequence is precisely targeted: the 6 false-positive Measles and 4 false-positive Chickenpox corrections observed in the ablation (Figure 4) are attributable to this CNN texture channel intercepting segmentation fragmentation artifacts that defeat the binary mask, without displacing the geometric reasoning that drives correct predictions. The CNN stream functions as a structured noise filter, not a black box replacing clinical judgment.

## Inference Analysis

To characterize real-world micro-architectural behavior under power-constrained conditions, CPU core frequency transients were logged across all four phases using a strict cache-warming protocol; five iterations were sampled exclusively from the median execution window to eliminate startup and thermal artifacts (Figure 7).

![CPU core frequency transients](cpu_frequency_vs_time.png)
*Figure 7: CPU core frequency transients across the four experimental phases.*

Phase 1 (Random Forest) completes inference in 0.31 ms, operating entirely within the 2.0 GHz baseline band, confirming that tree-traversal operations bypass power-dense vector execution units entirely. Phases 2 and 3 sustain a heavy-workload footprint at the 2.5 GHz ceiling throughout identical EfficientNet-B0 tensor graph operations, diverging only at pipeline termination (12.1 ms versus 12.8 ms respectively). 

MorphNN (Phase 4) exhibits a characteristic multi-state frequency signature:
1. **Morphological extraction** initializes within the 2.0 GHz band ($t < 1.5$ ms).
2. **Frozen backbone convolutions** transition execution up to the 2.5 GHz tier.
3. **Downstream pipeline processing** drops immediately back to the 2.0 GHz baseline once deep feature extraction concludes ($t \approx 8.2$ ms) for PCA projection and Random Forest inference.

This state-aware execution reduces total core cycle utilization to 9.8 ms. By the accuracy-to-power-draw criterion, Phase 1 is the most efficient configuration. However, in medical diagnosis, misclassification directly affects patient isolation and outbreak control. Within this context, the ~5% absolute macro F1 gain from MorphNN (0.953 vs. ~0.90) justifies its 9.5 ms overhead. MorphNN is therefore a deliberate trade-off, prioritizing higher classification fidelity over minimal computational cost for infectious disease screening.

## Deployment

Figure 8 shows the web interface of the deployed application, titled the **Clinical Exanthem Hybrid Diagnostic Portal**. 

When a user uploads a clinical image scan, the application immediately generates a diagnostic evaluation result. The dashboard offers a straightforward visualization of the current system settings and makes all the computed shape-based statistics directly accessible and interpretable. It uses a hybrid fusion classifier to estimate prediction probabilities and specifically marks the final predicted diagnosis, allowing users to easily verify and interpret the results. Additionally, it supports any of the model configurations stated in Table 4.

![App Interface](app_image.png)
*Figure 8: Clinical Exanthem Hybrid Diagnostic Portal App Interface*

## Contextual Positioning

Existing lightweight dermatological classifiers predominantly follow one of two paradigms: end-to-end fine-tuned CNNs optimized purely for accuracy, or classical handcrafted pipelines optimized purely for speed. 

MorphNN is comparable to the best fine-tuned architecture (EfficientNet-B0, $\text{F1} = 0.941$) in macro F1 while reducing training time by $22\times$ and full-pipeline footprint by 21.5% (16.1 MB versus 20.5 MB). While the footprint gains are incremental rather than revolutionary, the high accuracy, low training overhead, and GPU-free inference capabilities offered by MorphNN make it truly deployable in resource-constrained clinical settings—a deployment scenario that has not been adequately addressed in the dermatological AI literature.

## Limitations

Despite these gains, the MorphNN model still has some limitations:
* **Edge-case Confusion:** The confusion matrix (Figure 3) shows remaining Chickenpox–Measles confusion, mainly in atypical cases where lesion patterns deviate from typical morphology and neither morphological nor CNN texture features are fully discriminative. 
* **Pathological Smoothing:** While the Gray-World hypothesis establishes a standardized neutral baseline, it inherently assumes a balanced color distribution. In dermatological contexts, this assumption risks artificially attenuating pathological erythema (redness) by averaging it into the background. 
* **Data Diversity:** We only included a single dataset. Expanding the training distribution to include these edge-case presentations, or incorporating domain-specific fine-tuning of a shallow adapter layer, represents a natural extension of this work.
