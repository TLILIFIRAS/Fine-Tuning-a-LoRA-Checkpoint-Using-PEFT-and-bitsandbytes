# Fine-Tuning BLOOM with LoRA & PEFT using bitsandbytes

## Overview

This project demonstrates **efficient fine-tuning of large language models (LLMs)** using **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)** on a **BLOOM model**. By leveraging **8-bit quantization with bitsandbytes**, the project enables training and inference of large models with **reduced GPU memory and computation requirements**.

The model is fine-tuned on the **[Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)** dataset, allowing it to generate **quotes with contextual tags**. The workflow showcases an **end-to-end solution** for memory-efficient adaptation of massive language models.

---

## Project Goals

1. **Efficient Fine-Tuning**
   - Apply LoRA adapters to fine-tune a large model without updating all parameters.
   - Only a small subset of parameters (~0.1%) are trained, reducing computation and memory needs.

2. **Memory Optimization**
   - Use **bitsandbytes 8-bit quantization** for the base BLOOM model.
   - Enables training on a single GPU (e.g., 24GB) without sacrificing performance.

3. **Text Generation**
   - Fine-tune the model to generate **creative English quotes with associated tags**.
   - Maintain the base model's general knowledge while learning domain-specific patterns.

4. **Demonstrate Modern ML Tools**
   - Integrate Hugging Face **Transformers**, **PEFT**, **bitsandbytes**, and **Datasets**.
   - Show a practical workflow for training LLMs efficiently.

---

## Dataset

**Dataset Used:** [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes)

- **Format:** JSON with fields `quote` and `tags`.
- **Preprocessing:**  
  - Merge the `quote` and `tags` into a single text column using the format:  
    ```
    "<quote> ->: <tags>"
    ```
  - Tokenize the merged text for causal language modeling.

---

## Results

- The fine-tuned model generates **contextually relevant quotes**.
- Only a small fraction of parameters were trained via LoRA, demonstrating **parameter-efficient fine-tuning**.
- **8-bit quantization** allows training large models on modest GPU hardware.

---

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)  
- [PEFT](https://github.com/huggingface/peft)  
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)  
- [BLOOM model](https://huggingface.co/bigscience/bloom)
