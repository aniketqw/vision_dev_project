# AI Vision Reasoning Report
**Generated:** 2026-03-17 03:25  
**Training log:** `training_log_20260317_012558.json`  
**Misclassified log:** `misclassified_20260317_005331.json`  
**Model:** CIFAR-10 · 32×32 px · 3 epochs

---

## 1. Training Summary

| Metric | Value |
|--------|-------|
| Best Accuracy | **69.41%** |
| Best Epoch | 2 |
| Final Train Loss | 0.5882 |
| Final Val Loss | 0.8704 |
| Total Misclassified | 10012 |

**Epoch-level progression:**

| Epoch | Accuracy | Overall Acc | F1 | Misclassified |
|-------|----------|-------------|-----|---------------|
| 0 | 0.0000 | 0.6017 | 0.5937 | 4008 |
| 1 | 0.6050 | 0.6941 | 0.6916 | 3059 |
| 2 | 0.6941 | 0.7055 | 0.7092 | 2945 |

---

## 2. Failure Distribution by Distortion Type

| Distortion | Failures | % of Total | Avg Distortion Confidence |
|------------|---------|------------|--------------------------|
| **blur** | 1949 | 40.5% | 0.0382 |
| **jpeg** | 1434 | 29.8% | 0.0679 |
| **pixelate** | 708 | 14.7% | 0.0290 |
| **noise** | 105 | 2.2% | 0.0358 |
| **unknown** | 620 | 12.9% | 0.0000 |

---

## 3. Top Confusion Pairs (all distortions)

| True Class | Predicted Class | Count |
|-----------|----------------|-------|
| dog | cat | 220 |
| cat | dog | 214 |
| truck | automobile | 201 |
| automobile | truck | 176 |
| deer | frog | 163 |
| deer | horse | 161 |
| dog | horse | 154 |
| deer | bird | 149 |
| ship | airplane | 145 |
| airplane | ship | 132 |
| bird | frog | 125 |
| bird | airplane | 123 |
| cat | frog | 121 |
| bird | deer | 109 |
| horse | dog | 107 |

---

## 4. Per-Distortion Visual Analysis (Qwen2.5-VL)

> Each section below shows the model's blind spot analysis backed by
> direct visual inspection of representative misclassified images.

### 4.1 Blur — 1949 failures (40.5%)

**Most misclassified true classes:** cat (299), dog (259), bird (247), deer (215), airplane (185)  
**Most common wrong predictions:** airplane (282), cat (237), bird (216), truck (199), horse (193)  
**Avg distortion confidence:** 0.0382  
**Epoch distribution:** ep0:1515 · ep1:297 · ep2:137

#### Image 1 — `56cb777d94a395bc7eab6d1caed556e72dc75a44.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=0.289 |
| True class | **cat** |
| Predicted (wrong) | **dog** |
| Distortion confidence | `0.0228` |

**🔬 Distortion Artifacts Observed**

> Blur halos are visible around the edges of the image, particularly noticeable in the top-left and bottom-right corners. The overall image appears slightly out of focus, with a softening effect across the entire frame.

**✅ Surviving True-Class Features**

> The cat's silhouette shape is discernible despite the blur. The distinct head shape, pointed ears, and the general form of the body are still recognizable as cat-like features.

**❌ What Misled the Model**

> The blurred region in the center of the image, which appears to be the fur area, has a texture that resembles the pattern of a dog's fur. This texture, combined with the overall softness of the image, misled the model into predicting "dog."

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the general form and structure of the animal as cat-like, despite the blurring.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly predicted "dog" due to the blurred texture in the fur area resembling dog fur. This texture, combined with the overall softness of the image, triggered the wrong class activation.

**📊 Confidence Assessment**

> The distortion detection confidence of 0.0228 seems low because the blur is subtle and not immediately obvious. However, upon closer inspection, the image is not entirely clear, and the texture in the fur area is somewhat ambiguous, making the prediction less certain than it might appear.

**🎯 Root Cause**

> The single core reason why blur distortion on a cat image specifically triggers a dog prediction is the similarity between the blurred texture in the fur area and the texture typically associated with dog fur.

#### Image 2 — `3c9ab182a5013105d667c5e92e452b291c7f50fe.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=1.180 |
| True class | **dog** |
| Predicted (wrong) | **horse** |
| Distortion confidence | `0.0193` |

**🔬 Distortion Artifacts Observed**

> The blur distortion introduces a smooth, hazy appearance across the entire image. There are no distinct edges or sharp transitions, which obscures finer details of the subject's features.

**✅ Surviving True-Class Features**

> The image retains a general silhouette shape resembling a dog, with a darker area that could be interpreted as a head or face, and lighter areas that might represent fur. The overall structure of the body is discernible despite the blurring.

**❌ What Misled the Model**

> The blurred fur area near the top of the image resembles the mane of a horse, which the model may have incorrectly identified as part of the horse's body.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the general shape and structure of the subject as a quadrupedal animal.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly classified the subject as a horse due to the blurred fur area near the top resembling a horse's mane. This misinterpretation led the model to associate the image with a horse rather than a dog.

**📊 Confidence Assessment**

> The distortion confidence of 0.0193 seems low given the subtle nature of the blur. Despite the blurring, the image contains enough recognizable features to suggest a dog, but the model's misclassification indicates that the distortion significantly altered the perception of certain features.

**🎯 Root Cause**

> The blur distortion made the fur area near the top of the image appear like a horse's mane, leading the model to predict a horse instead of a dog.

#### Image 3 — `782de3f7d0305d2701d72cf6cb02dd8e55b674d3.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=1.236 |
| True class | **dog** |
| Predicted (wrong) | **ship** |
| Distortion confidence | `0.0192` |

**🔬 Distortion Artifacts Observed**

> Blur halos are visible around the edges of the image, particularly noticeable around the dog's head and ears. The blur introduces a soft, hazy effect across the entire image, obscuring finer details.

**✅ Surviving True-Class Features**

> The dog's silhouette shape is discernible despite the blur. The overall form of the dog, including its head and body, is recognizable as a dog. The color blobs, such as the white fur and darker areas, are still present but less defined due to the blur.

**❌ What Misled the Model**

> The blurred edges of the dog's head and ears resemble the angular shapes of a ship's bow and hull. The soft, rounded edges of the dog's fur blend into the background, creating a pattern that the model misinterprets as the straight lines and sharp angles characteristic of a ship.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified some features of the dog, such as the general silhouette and color patterns, which were partially preserved despite the blur.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly predicted "ship" because the blurred edges of the dog's head and ears matched the visual pattern of a ship's bow and hull. The soft, rounded edges of the dog's fur blended into the background, creating a pattern that the model misinterpreted as the straight lines and sharp angles of a ship.

**📊 Confidence Assessment**

> The distortion detection confidence of 0.0192 seems low given the clear visual cues that the blur introduces. Despite the subtle nature of the distortion, the image is not ambiguous; the visual pattern of the dog's fur edges is unmistakably similar to a ship's bow and hull, making the prediction of "ship" highly plausible.

**🎯 Root Cause**

> The single core reason for the blur distortion on a dog image triggering a ship prediction is the similarity between the blurred edges of the dog's head and ears and the angular shapes of a ship's bow and hull.

### 4.2 Jpeg — 1434 failures (29.8%)

**Most misclassified true classes:** deer (256), bird (214), cat (201), dog (183), horse (128)  
**Most common wrong predictions:** frog (207), horse (175), bird (168), cat (161), deer (158)  
**Avg distortion confidence:** 0.0679  
**Epoch distribution:** ep0:1156 · ep1:188 · ep2:90

#### Image 1 — `b8e50fbf6593ad99c6d379e62c630fc809c8ef45.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=0.793 |
| True class | **cat** |
| Predicted (wrong) | **deer** |
| Distortion confidence | `0.0369` |

**🔬 Distortion Artifacts Observed**

> The image exhibits noticeable JPEG compression artifacts, particularly in the fur area and the background. The top-left corner shows a slight blurring effect, while the background appears as a series of small, square blocks, indicative of the JPEG 8x8 block structure.

**✅ Surviving True-Class Features**

> The cat's distinctive whiskers and the texture of the fur are still visible despite the compression. The white fur and the pinkish hue of the nose are clear, suggesting the image retains some true-class features.

**❌ What Misled the Model**

> The model was misled by the blurred and distorted areas in the fur, which could be interpreted as the antlers of a deer. The background blocks might have been mistaken for the deer's body.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the white fur and the pink nose as part of a cat.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly classified the image as a deer due to the blurred and distorted fur areas being misinterpreted as antlers. The background blocks were also likely confused with the deer's body.

**📊 Confidence Assessment**

> The distortion detection confidence of 0.0369 seems low because the image contains subtle artifacts that are not immediately obvious but can be discerned upon close inspection. A human observer would likely notice the JPEG compression and understand the potential misclassification risk.

**🎯 Root Cause**

> The single core reason why JPEG distortion on a cat image specifically triggers a deer prediction is the misinterpretation of the blurred fur areas as antlers.

#### Image 2 — `41000dc8a8f7969e4d424527892b78916a2bda2d.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=1.655 |
| True class | **horse** |
| Predicted (wrong) | **dog** |
| Distortion confidence | `0.0499` |

**🔬 Distortion Artifacts Observed**

> The image exhibits noticeable blurring, particularly around the edges of the horse's head and body, indicative of JPEG compression artifacts. The blurring is more pronounced in the top-left and bottom-right corners, where the JPEG block boundaries are less defined.

**✅ Surviving True-Class Features**

> The horse's silhouette shape, characterized by a distinct head and neck, is still recognizable despite the blurring. The color blobs, such as the brown mane and the white areas around the face, also survive the distortion, albeit with some loss of definition.

**❌ What Misled the Model**

> The model was misled by the blurred regions in the top-left corner, which could be interpreted as a dog's head due to the similar shape and color gradient. This area, which should have been the horse's ear or part of the mane, now resembles a dog's facial features.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the overall silhouette as a horse, despite the blurring.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly classified the image as a dog because of the blurred region in the top-left corner, which it misinterpreted as a dog's head. The color and shape similarities between the blurred area and a dog's face triggered the wrong classification.

**📊 Confidence Assessment**

> The distortion detection confidence of 0.0499 suggests that the distortion is subtle but noticeable. While the image is not entirely ambiguous to a human eye, the blurring and color gradients in the top-left corner are significant enough to cause confusion for the model.

**🎯 Root Cause**

> JPEG compression artifacts in the top-left corner of the image caused the model to misinterpret the blurred region as a dog's head, leading to the incorrect classification.

#### Image 3 — `1235a61c0966cb017223c1bbb8a1adbd7da91235.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=1.927 |
| True class | **cat** |
| Predicted (wrong) | **deer** |
| Distortion confidence | `0.0153` |

**🔬 Distortion Artifacts Observed**

> The image exhibits noticeable JPEG block artifacts, particularly evident in the fur areas and edges of the cat's face. The blockiness is most pronounced in the top-left and bottom-right corners, where the pixels appear to be grouped into larger, less defined blocks rather than smooth transitions.

**✅ Surviving True-Class Features**

> The cat's distinct facial features, such as the eyes and whiskers, are still recognizable despite the distortion. The fur texture, while somewhat mottled due to the JPEG compression, retains some of the characteristic patterns of a cat's coat.

**❌ What Misled the Model**

> The model likely misinterpreted the blurred and distorted edges of the cat's face as the antlers of a deer. The softened, rounded shapes in the top-left corner of the image, which were originally part of the cat's fur, now resemble the antlers of a deer.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the cat's eyes and the general shape of the head, which are still discernible despite the JPEG distortion.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly classified the image as a deer because it misread the blurred edges of the cat's face as antlers. This misreading was triggered by the JPEG block artifacts, which made the edges appear more defined and angular than they actually were.

**📊 Confidence Assessment**

> The distortion detection confidence of 0.0153 suggests that the distortion is subtle but noticeable. While the image is not entirely ambiguous to a human eye, the subtle nature of the JPEG artifacts could lead to misinterpretation, especially for a model that relies heavily on edge detection.

**🎯 Root Cause**

> The JPEG block artifacts caused the model to misinterpret the cat's fur edges as the antlers of a deer, leading to the incorrect classification.

### 4.3 Pixelate — 708 failures (14.7%)

**Most misclassified true classes:** deer (126), bird (112), cat (111), dog (104), frog (58)  
**Most common wrong predictions:** frog (107), horse (99), deer (86), bird (83), cat (67)  
**Avg distortion confidence:** 0.0290  
**Epoch distribution:** ep0:580 · ep1:87 · ep2:41

#### Image 1 — `86704b4a920a803d4a5fd7cece78c2840c4ce3be.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=0.569 |
| True class | **cat** |
| Predicted (wrong) | **dog** |
| Distortion confidence | `0.0240` |

**🔬 Distortion Artifacts Observed**

> Pixelate distortion introduces a grid of 8x8 squares across the entire image, obscuring finer details and creating a blocky appearance.

**✅ Surviving True-Class Features**

> The cat's fur texture, particularly in the top-left region, still shows some fine details despite the pixelation. The overall shape and structure of the head and ears are recognizable as belonging to a cat.

**❌ What Misled the Model**

> The pixelated background, which looks like a light-colored wall, resembles the fur of a dog due to the uniform color and lack of distinct features.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the cat's fur texture and shape as part of the true-class features.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly classified the image as a dog because the pixelated background, which lacks distinct features, was mistaken for dog fur. This is due to the uniformity and light color of the background, which can be easily confused with the fur of a dog.

**📊 Confidence Assessment**

> The low distortion confidence (0.0240) suggests that the pixelation is subtle but noticeable. While the image is not entirely ambiguous to a human eye, the model's misclassification indicates that the distortion is significant enough to alter the perception of the image.

**🎯 Root Cause**

> The pixelate distortion creates a uniform background that closely resembles the fur of a dog, leading the model to predict "dog" instead of "cat."

#### Image 2 — `95ecdba043d2d673e7f88a9463ca06e3fbcbab09.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=2.717 |
| True class | **cat** |
| Predicted (wrong) | **frog** |
| Distortion confidence | `0.0536` |

**🔬 Distortion Artifacts Observed**

> Pixelate distortion introduces a grid of 8x8 squares across the entire image, obscuring finer details and creating a blocky appearance.

**✅ Surviving True-Class Features**

> The cat's eyes retain their circular shape and color gradient, which are key features suggesting the true class. The fur texture, though blurred, still shows the characteristic patterns of a cat.

**❌ What Misled the Model**

> The frog-like appearance arises from the pixelated grid pattern in the fur area, which the model may have interpreted as the smooth, uniform texture of a frog's skin.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the cat's eyes as belonging to a cat, despite the pixelation.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly classified the image as a frog due to the pixelated grid pattern in the fur area, which the model misinterpreted as frog skin texture.

**📊 Confidence Assessment**

> The low distortion confidence (0.0536) suggests that the pixelation is subtle but noticeable enough to affect the model's decision-making process, making the classification uncertain even for humans.

**🎯 Root Cause**

> The pixelated grid pattern in the fur area created by the distortion resembles the smooth texture of a frog's skin, leading the model to predict "frog."

#### Image 3 — `eda67c820e1ce668bfed71972fbe9a40cb709614.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=2.998 |
| True class | **cat** |
| Predicted (wrong) | **dog** |
| Distortion confidence | `0.0462` |

**🔬 Distortion Artifacts Observed**

> Pixelation introduces a grid of 8x8 squares across the entire image, obscuring finer details and creating a blocky appearance.

**✅ Surviving True-Class Features**

> The overall shape of the head and ears, along with the fur texture, still suggest a cat. The eyes and nose maintain some recognizable features despite the pixelation.

**❌ What Misled the Model**

> The pixelated background, which looks like a light-colored wall, resembles the white fur of a dog's chest. This area, combined with the blocky texture, may have misled the model into predicting a dog.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the general shape and fur texture as belonging to a cat.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly predicted a dog due to the pixelated background resembling a dog's chest. The blocky texture and light color of the background triggered the wrong class activation.

**📊 Confidence Assessment**

> The low distortion confidence (0.0462) suggests that while the pixelation is noticeable, it is not entirely obvious to a human eye. The image remains somewhat ambiguous, but the model's prediction is not strongly supported by the distorted features.

**🎯 Root Cause**

> The pixelated background, resembling a dog's chest, triggered the model to predict a dog due to the similarity in color and texture.

### 4.4 Noise — 105 failures (2.2%)

**Most misclassified true classes:** cat (19), bird (17), dog (12), ship (12), truck (11)  
**Most common wrong predictions:** cat (16), airplane (16), dog (15), horse (11), automobile (10)  
**Avg distortion confidence:** 0.0358  
**Epoch distribution:** ep0:93 · ep1:8 · ep2:4

#### Image 1 — `ead1f465a5bd548e04a691c1d9e412a41a7cf46e.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=1.528 |
| True class | **cat** |
| Predicted (wrong) | **frog** |
| Distortion confidence | `0.0567` |

**🔬 Distortion Artifacts Observed**

> Noise speckle pattern is visible throughout the image, particularly noticeable as small white dots scattered across the cat's fur and the background.

**✅ Surviving True-Class Features**

> The cat's distinct black and white fur pattern is partially preserved, especially in the top-left and bottom-right areas of the image. The overall shape of the cat's head and body is recognizable despite the noise.

**❌ What Misled the Model**

> The noise speckles in the fur area near the center of the image create a pattern that resembles the eyes and nose of a frog, misleading the model into predicting "frog."

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the general shape and coloration of the cat, which are still discernible despite the noise.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly predicted "frog" due to the noise speckles in the fur area resembling the eyes and nose of a frog. This pattern is visually similar to the frog's features, triggering the wrong class activation.

**📊 Confidence Assessment**

> The distortion detection confidence of 0.0567 seems low given the subtle nature of the noise speckles. While the image is not entirely ambiguous, the noise is not immediately obvious to a human eye, suggesting the model's confidence should be higher than indicated.

**🎯 Root Cause**

> The noise speckles in the cat's fur area create a pattern that closely resembles the eyes and nose of a frog, leading to the incorrect prediction.

#### Image 2 — `756a90209c442dc89adb7ccf80936298514a8f08.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=2.160 |
| True class | **cat** |
| Predicted (wrong) | **truck** |
| Distortion confidence | `0.0124` |

**🔬 Distortion Artifacts Observed**

> The noise distortion introduces a speckle pattern across the entire image, with varying intensities of white and gray dots. The speckles are distributed randomly but cover most of the image, particularly noticeable in the darker areas such as the cat's fur and the background.

**✅ Surviving True-Class Features**

> The cat's fur texture, especially in the lighter areas, still shows some distinct color blobs that resemble the fur pattern of a cat. The overall shape of the head and body also suggests a cat, though heavily distorted.

**❌ What Misled the Model**

> The speckle pattern in the darker regions of the image, particularly around the cat's eyes and nose, creates a visual pattern that resembles the headlights and front grille of a truck. This pattern is more pronounced in the bottom left corner of the image.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the general shape and structure of the cat, despite the heavy distortion.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly activated for the truck class due to the speckle pattern in the darker regions resembling the headlights and front grille of a vehicle. This pattern is visually similar to the features typically associated with trucks.

**📊 Confidence Assessment**

> The distortion detection confidence of 0.0124 seems low given the significant visual changes introduced by the noise. However, the model's prediction is still plausible because the distortion is subtle enough to be overlooked by humans, yet it is enough to mislead the model into a wrong classification.

**🎯 Root Cause**

> The random speckle pattern in the darker regions of the image, which resembles the headlights and front grille of a truck, triggered the model's activation for the truck class.

#### Image 3 — `d599d49df5be108e1a4b5ef63e456759add30308.png`

| Field | Value |
|-------|-------|
| Role | ⭕ **typical** · cluster dist=3.386 |
| True class | **deer** |
| Predicted (wrong) | **automobile** |
| Distortion confidence | `0.0108` |

**🔬 Distortion Artifacts Observed**

> The noise distortion introduces a speckle pattern across the entire image, particularly noticeable as small white dots scattered throughout the deer's fur and the background.

**✅ Surviving True-Class Features**

> The deer's silhouette shape, especially the antlers and the overall body structure, remains recognizable despite the noise. The color blobs of the fur are still discernible, albeit less sharp due to the distortion.

**❌ What Misled the Model**

> The noise speckles in the fur area create a pattern that resembles the headlights and grille of an automobile, which the model incorrectly associates with the image.

**🧠 Model Reasoning — Correct Part**

> The model correctly identified the general shape of the deer, including the antlers and the body structure.

**🚫 Model Reasoning — Incorrect Part**

> The model incorrectly predicted automobile due to the speckle pattern in the fur area resembling the headlights and grille of a car. This pattern is visually similar to the features typically associated with automobiles.

**📊 Confidence Assessment**

> The distortion detection confidence of 0.0108 seems low given the obvious speckle pattern that significantly alters the image. A human observer would likely notice the distortion easily, suggesting the model's confidence should be higher for such a clear misclassification.

**🎯 Root Cause**

> The speckle pattern in the fur area creates a visual similarity to the headlights and grille of an automobile, leading the model to predict the wrong class.

---

## 5. Model Blind Spot Summary

| Distortion | Visual Blind Spot (synthesised from VLM analysis) |
|------------|--------------------------------------------------|
| **blur** | The single core reason why blur distortion on a cat image specifically triggers a dog prediction is the similarity between the blurred texture in the fur area and the texture typically associated with dog fur. \| The blur distortion made the fur area near the top of the image appear like a horse's mane, leading the model to predict a horse instead of a dog. \| The single core reason for the blur distortion on a dog image triggering a ship prediction is the similarity between the blurred edges of the dog's head and ears and the angular shapes of a ship's bow and hull. |
| **jpeg** | The single core reason why JPEG distortion on a cat image specifically triggers a deer prediction is the misinterpretation of the blurred fur areas as antlers. \| JPEG compression artifacts in the top-left corner of the image caused the model to misinterpret the blurred region as a dog's head, leading to the incorrect classification. \| The JPEG block artifacts caused the model to misinterpret the cat's fur edges as the antlers of a deer, leading to the incorrect classification. |
| **pixelate** | The pixelate distortion creates a uniform background that closely resembles the fur of a dog, leading the model to predict "dog" instead of "cat." \| The pixelated grid pattern in the fur area created by the distortion resembles the smooth texture of a frog's skin, leading the model to predict "frog." \| The pixelated background, resembling a dog's chest, triggered the model to predict a dog due to the similarity in color and texture. |
| **noise** | The noise speckles in the cat's fur area create a pattern that closely resembles the eyes and nose of a frog, leading to the incorrect prediction. \| The random speckle pattern in the darker regions of the image, which resembles the headlights and front grille of a truck, triggered the model's activation for the truck class. \| The speckle pattern in the fur area creates a visual similarity to the headlights and grille of an automobile, leading the model to predict the wrong class. |

---

## 6. Actionable Recommendations

Based on the failure distribution and VLM visual analysis above:

| Priority | Recommendation | Addresses |
|----------|---------------|-----------|
| 🔴 High | **Multi-distortion augmentation** — add blur (σ=0.5–3), JPEG sim (q=10–70), pixelation (block 2–8px), Gaussian noise (σ=0.05–0.25) to training dataloader | Blur (40%), JPEG (30%), Pixelate (15%) |
| 🔴 High | **Dual-pathway backbone** — add a low-pass feature stream (shape/silhouette) alongside the standard high-freq stream; merge with channel attention (SE-Net/CBAM) | Blur, Pixelate |
| 🟡 Medium | **Gradient-reversal distortion-adversarial head** — auxiliary head predicts distortion type with GRL; forces backbone to learn distortion-invariant features | All types |
| 🟡 Medium | **Fine-grained contrastive loss** — add SupCon or NT-Xent loss for the cat/dog/deer/bird confusion cluster; enforce larger embedding margins between visually similar classes | JPEG, Blur |
| 🟢 Low | **Train longer + cosine LR decay** — current 3-epoch run shows monotonically falling loss; model has not converged. Val loss > train loss indicates early overfitting | All types |

---

*Report generated by `pipe/vision_reasoning_report.py` · 2026-03-17 03:25*