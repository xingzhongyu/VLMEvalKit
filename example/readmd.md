# Evaluation Guide: Industrial Manufacturing VQA Dataset

Thank you for participating in the evaluation of our Visual Question Answering (VQA) dataset for the manufacturing industry. Your domain expertise is crucial to ensuring the accuracy, reliability, and logical soundness of this data.

## Project Overview
This spreadsheet contains a dataset of industrial images paired with multiple-choice questions, correct answers, and explanations (justifications). Your task is to review each QA pair alongside its image and rate its quality across five key dimensions using a 1-5 scoring scale.

## Instructions
Please evaluate each row in the dataset by assigning a score from 1 (Lowest/Unacceptable) to 5 (Highest/Excellent) in the designated scoring columns. 

If you assign a score of 3 or lower in any category, please briefly explain the issue or provide the correct phrasing in the **"Expert Comments / Suggested Fixes"** column.

### Evaluation Rubric (1-5 Scale)

#### 1. Domain Accuracy
Evaluates if the provided `answer` is factually correct according to strict industry standards.
* **5:** Absolutely correct. Uses precise, standard industrial terminology.
* **4:** Correct, but terminology could be slightly more professional.
* **3:** Partially correct or debatable depending on niche contexts.
* **2:** Contains noticeable technical errors.
* **1:** Completely incorrect or violates fundamental engineering/manufacturing principles.

#### 2. Visual Answerability
Evaluates whether the question can actually be answered by looking at the provided image.
* **5:** The visual features required to answer the question are clear and unambiguous in the image.
* **4:** Can be answered from the image, but requires slight zooming or inference.
* **3:** The image is somewhat ambiguous or low quality, making the answer a slightly educated guess.
* **2:** Key features are highly obscured or missing from the visual frame.
* **1:** Impossible to answer from the image alone (e.g., asking for the internal voltage of a closed, opaque cabinet).

#### 3. Distractor Quality
Evaluates the quality of the incorrect options (the choices not listed in the `answer` column).
* **5:** Distractors are highly plausible and represent common misconceptions or closely related industrial concepts.
* **4:** Good distractors, but one or two might be slightly obvious to rule out.
* **3:** Distractors are average; functional but not particularly challenging.
* **2:** Most distractors are completely unrelated to the context of the image.
* **1:** Distractors are nonsensical, obvious "throwaways," or logically eliminate themselves.

#### 4. Question Clarity
Evaluates the phrasing and lack of ambiguity in the `question`.
* **5:** The question is direct, professional, and unambiguous. If multiple answers are required, the phrasing implies it.
* **4:** Clear, but phrasing could be slightly tightened.
* **3:** A bit vague (e.g., "What does this image show?"), leaving room for multiple valid interpretations.
* **2:** Confusing or grammatically poor, making it difficult to understand the core inquiry.
* **1:** Incomprehensible or contradictory.

#### 5. Justification Quality
Evaluates the usefulness of the explanation provided in the `justification` column.
* **5:** Excellent. It explicitly connects the correct answer to specific visual evidence in the image (e.g., "Identifiable by the metallic luster and specific mounting holes...").
* **4:** Good explanation, but misses pointing out one or two visual cues.
* **3:** Merely states facts without tightly connecting them to the image.
* **2:** Simply repeats the answer without explaining *why*.
* **1:** The justification is incorrect, contradictory, or empty.

## Quick Tips
* **Multiple Answers:** Some questions have multiple correct answers (e.g., `E;G`). Please ensure both the question phrasing and justification support a multi-select format.
* **Focus on the Core:** The most critical metrics are **Domain Accuracy** and **Visual Answerability**. If a question fails either of these, the data point is unusable.