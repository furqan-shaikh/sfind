# sfind 🕵️‍♂️

**Tagline:** find, but semantic.

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/sfind.svg)](https://pypi.org/project/sfind/)
[![GitHub stars](https://img.shields.io/github/stars/<your-username>/sfind?style=social)](https://github.com/<your-username>/sfind)

---

## 🚀 What is sfind?

Stop searching by filenames. **sfind** lets you search your filesystem by **meaning and content**. Describe what you want — sfind finds it.  

- 🔹 **Semantic search** using AI models like **Perception Encoder** & **CLIP**  
- 🔹 **Filesystem as a vector store**: embeddings stored in file inodes (xattrs)  
- 🔹 **Explainable image results**: captions for image files  
- 🔹 **Fast CLI interface** with ranked results  

---

## ⚡ Quick Start

```bash
git clone <repo-url>
cd sfind
pip install -e .
```

## Search example
`sfind --query "novak djokovic playing tennis" --describe-image`

Results for query: novak djokovic playing tennis
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃ File                           ┃ Score ┃ Caption                         ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │ /path/to/image1.jpg            │ 0.311 │ a tennis player in action on a green court │
│ 2 │ /path/to/file2.pdf             │ 0.198 │ a certificate for a child      │
│ 3 │ /path/to/file3.txt             │ 0.186 │ a letter from the owner of the home │

## 🔧 How it works
 - Embedding generation: files are encoded into vectors by your chosen model.
 - Stored in filesystem: embeddings saved as extended attributes (xattrs) in the inode.
 - Query & rank: semantic query matched against stored embeddings.
 - Explainable captions: images get a caption describing why they match.

## 🤖 Supported Models

| Name                      | Type             | Description                                                                  | Model ID            | Size |
|---------------------------|------------------|------------------------------------------------------------------------------|------------------|------|
| Meta's Perception Encoder | Vision Encoder   | Family of the state-of-the-art vision encoders for encoding images and video | PE-Core-L14-336  | 2.68 GB
| CLIP                      | Vision Encoder   | OpenAI's versatile embeddings for visual & textual data                      | openai/clip-vit-base-patch32 | 605 MB
| SalesForce BLIP           | Captioning Model | Salesforce’s BLIP (Bootstrapping Language-Image Pre-training) model                                 | Salesforce/blip-image-captioning-base   | 990 MB



## 📜 License
MIT License


