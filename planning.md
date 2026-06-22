# TakeMeter Planning

## Project focus

TakeMeter will classify discourse quality and function in the public **Stack Overflow `python` tag** community, where beginners and experienced programmers discuss Python questions, answers, and clarifying comments. This community is a good fit because posts vary widely: some users provide detailed debugging context, some answers explain concepts carefully, some replies give terse fixes, and some comments are mostly reactions or low-context follow-ups. These distinctions matter because useful technical discussion depends not only on whether an answer is correct, but whether it gives enough context for learners to understand and act on it.

## Label taxonomy

The classifier will use four mutually exclusive labels. The labels are designed for individual posts or comments, not entire threads.

### `diagnostic_request`

A post or comment asking for help with a Python problem while providing enough concrete information to diagnose it, such as code, an error message, expected behavior, observed behavior, or a specific technical question.

Clear examples:

1. "I'm trying to read a CSV with pandas, but `read_csv` treats the first row as data instead of headers. My file starts with `name,age`, and this is my code: `pd.read_csv('people.csv', header=None)`. Why is the header not being detected?"
2. "This recursive function returns `None` after the first call. I expected it to return the final list. What am I missing? [code omitted here for planning example]"

Boundary example:

- "Why doesn't my code work?" This is asking for help, but it lacks diagnostic information. I will label this `low_context_reaction` unless the surrounding text includes concrete details.

### `explanatory_answer`

A response that teaches or explains a Python concept, debugging principle, or reasoning process, not merely the final answer.

Clear examples:

1. "The issue is that lists are mutable, so using `[]` as a default argument means every call shares the same object. Use `None` as the default and create a new list inside the function."
2. "`range(10)` stops before 10 because Python ranges are half-open. This makes loops line up with zero-based indexing and length calculations."

Boundary example:

- "Use `enumerate(items)` instead." This may be correct, but without reasoning it is a `direct_fix`, not an `explanatory_answer`.

### `direct_fix`

A concise answer that gives a code snippet, command, API name, or direct correction with little or no explanation of why it works.

Clear examples:

1. "Use `json.loads(response.text)` instead of `json.load(response.text)`."
2. "Change `if x = 3:` to `if x == 3:`."

Boundary example:

- "Use `Path.read_text()` because it handles opening and closing the file for you." This includes reasoning, so I will label it `explanatory_answer` rather than `direct_fix`.

### `low_context_reaction`

A post or comment that is mostly emotional reaction, social feedback, vague complaint, joke, thanks, meta-commentary, or a help request without enough technical context to diagnose.

Clear examples:

1. "Python is so confusing, I give up."
2. "Thanks, that fixed it!"

Boundary example:

- "I'm totally stuck with this loop; it never ends, and here is the code..." Despite the emotional opening, the included code makes it a `diagnostic_request`.

## Decision rules for hard edge cases

1. **Answer with both code and explanation:** If the response includes a reason, tradeoff, or conceptual explanation that would help the learner generalize beyond the immediate fix, label it `explanatory_answer`. If it only gives the changed line or command, label it `direct_fix`.
2. **Question with frustration:** Emotional language does not decide the label. If the post includes code, error text, expected behavior, or a specific Python concept, label it `diagnostic_request`. If it only expresses frustration or asks a vague question, label it `low_context_reaction`.
3. **Very short but specific question:** A short post can still be `diagnostic_request` if it names a concrete problem, such as "Why does `list.sort()` return None?" If the question cannot be answered without guessing, label it `low_context_reaction`.
4. **Meta-advice about learning Python:** If the comment explains a learning strategy with reasoning, label it `explanatory_answer`. If it is only encouragement or opinion, label it `low_context_reaction`.

## Anticipated difficult cases

These are the cases I expect to revisit during annotation.

| Example | Possible labels | Decision |
| --- | --- | --- |
| "Use a virtual environment." | `direct_fix`, `explanatory_answer` | `direct_fix` unless the comment explains dependency isolation or reproducibility. |
| "Can someone explain classes? I don't get them." | `diagnostic_request`, `low_context_reaction` | `low_context_reaction` because the request is too broad and lacks a specific problem. |
| "Your indentation is wrong; Python uses indentation to define blocks." | `direct_fix`, `explanatory_answer` | `explanatory_answer` because it states the underlying rule. |

After collecting the dataset, I will replace or extend this table with at least three real difficult-to-label examples from the data.

## Data collection plan

- **Source:** Public Stack Overflow content from the `python` tag, collected through the Stack Exchange public API.
- **Unit of analysis:** One question title/body, one answer, or one comment.
- **Minimum size:** 200 labeled examples.
- **Current dataset size:** 220 examples in `data/labeled_examples.csv`.
- **Current distribution:** 55 examples per label, so no label dominates the dataset.
- **Collection approach:** Use `scripts/collect_stackoverflow_python.py` to collect public text and source URLs. I will avoid usernames and unnecessary personal information in the model input.
- **Dataset format:** A single CSV file with these columns:
  - `text`
  - `label`
  - `notes`
  - `source_url`
- **Important annotation note:** The generated CSV uses weak heuristic labels based on Stack Overflow post type and text cues. Before final submission, I need to manually review and correct the labels, especially comments labeled `low_context_reaction` and short answers labeled `direct_fix`.

## Annotation process

1. Read each example fully before assigning a label.
2. Apply the decision rules above rather than relying on overall impression.
3. Use the `notes` column for examples that are ambiguous, unusually short, sarcastic, or dependent on missing thread context.
4. After the first 30–40 examples, check whether any label definition needs tightening before annotating the full dataset.
5. After all examples are labeled, compute label counts and rebalance if one label dominates.

## Evaluation metrics

Accuracy is useful because the labels are mutually exclusive and each example receives exactly one label. However, accuracy alone can hide failures on smaller classes, so I will also report **precision, recall, and F1 for each label**.

Per-class F1 is especially important for this task because `direct_fix` and `explanatory_answer` may be easy to confuse. If the model performs well overall but has low F1 on one of those labels, it means the classifier is not learning the discourse distinction I care about. I will also include a confusion matrix to identify directional errors, such as `direct_fix` being over-predicted when the true label is `explanatory_answer`.

## Definition of success

For this classifier to be genuinely useful as a lightweight community-analysis tool, I would want:

- Fine-tuned model accuracy of at least **0.70** on the test set.
- Per-class F1 of at least **0.60** for every label.
- Fine-tuned model accuracy at least **10 percentage points higher** than the zero-shot Groq baseline, or a clearly better per-class balance if overall accuracy is similar.

For a real deployed moderation or recommendation tool, this would not be enough on its own. I would want a larger dataset, multiple annotators, clearer privacy handling, and calibration checks before making user-facing decisions from the model.

## Baseline plan

The zero-shot baseline will use Groq's `llama-3.3-70b-versatile` on the locked test set. The prompt will include the four label definitions and instruct the model to output only one of:

- `diagnostic_request`
- `explanatory_answer`
- `direct_fix`
- `low_context_reaction`

I will record the exact prompt in the README after running the baseline.

## Fine-tuning plan

The fine-tuned model will start from `distilbert-base-uncased` using the provided Colab notebook. I plan to begin with the notebook defaults:

- Epochs: 3
- Learning rate: `2e-5`
- Train batch size: 16
- Evaluation batch size: 32
- Max token length: 256

The initial reason for keeping 3 epochs is that the dataset is small, so additional epochs could overfit the training examples rather than improving generalization.

## AI Tool Plan

### Label stress-testing

I will ask an AI tool to generate 5–10 borderline Stack Overflow Python-style examples that sit between these pairs:

- `direct_fix` vs. `explanatory_answer`
- `diagnostic_request` vs. `low_context_reaction`
- `explanatory_answer` vs. `low_context_reaction`

If generated examples cannot be classified cleanly using the decision rules, I will revise the label definitions before annotating the full dataset.

### Annotation assistance

I used a script to weak-label collected examples based on source type and text cues, rather than using an LLM as the sole annotator. Every example still needs manual review before the dataset should be treated as final ground truth. I will document this weak-labeling workflow in the README's AI/data usage section and will not treat script-generated labels as automatically correct.

### Failure analysis

After fine-tuning, I will give the model's wrong predictions to an AI tool and ask it to identify possible error patterns, such as short posts, missing thread context, or confusion between direct fixes and explanations. I will verify any suggested patterns myself by rereading the examples and checking the confusion matrix before writing the evaluation report.

## Stretch features considered

I am not starting a stretch feature yet. If I add one later, I will update this section before beginning it.

Potential stretch options:

- **Error pattern analysis:** Most likely, because it directly improves the README evaluation.
- **Confidence calibration:** Useful if the notebook exposes prediction probabilities clearly.
- **Deployed interface:** Possible after the core report is complete.
