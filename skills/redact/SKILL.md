---
name: redact
description: |
  Privacy redaction toolkit for images, PDFs, Word documents, and PowerPoint presentations. 
  Use when the user needs to redact, mask, or replace sensitive/private information in files.
  
  Triggers:
  - Redacting or masking sensitive text in images, PDFs, documents, or presentations
  - Replacing names, phone numbers, IDs, or other PII in files
  - Processing privacy compliance for documents before sharing
  - Anonymizing content in visual files
  
  Supported formats: png/jpg images, PDF, docx/doc, pptx/ppt
---

# Redact Skill

Privacy redaction toolkit using PPStructureV3 OCR for text detection and replacement.

## Scripts

| Script | Format | Command |
|--------|--------|---------|
| `redact-image.py` | Images (png, jpg, etc.) | `redact-image.py <input> <rules.csv> <output>` |
| `redact-pdf.py` | PDF | `redact-pdf.py <input> <rules.csv> <output>` |
| `redact-document.py` | Word (docx, doc) | `redact-document.py <input> <rules.csv> <output>` |
| `redact-presentation.py` | PowerPoint (pptx, ppt) | `redact-presentation.py <input> <rules.csv> <output>` |

## CSV Rules Format

```csv
target_text,replacement_text
张三,李四
手机号,
身份证号,
```

| Rule | Effect |
|------|--------|
| `原文本,新文本` | Replace with new text |
| `原文本,` | Empty = mask with █ (documents) or solid color block (images/PDF) |

## Masking Behavior

| Format | Empty Replacement |
|--------|-------------------|
| Images, PDF | Solid color block overlay |
| Word, PowerPoint | `█` characters (same length as target) |

## Features

| Feature | Image | PDF | Document | Presentation |
|---------|:-----:|:---:|:--------:|:------------:|
| Text replacement | ✅ | ✅ | ✅ | ✅ |
| Solid color mask | ✅ | ✅ | - | - |
| █ character mask | - | - | ✅ | ✅ |
| OCR detection | ✅ | ✅ | ✅ (images) | ✅ (images) |
| Tables | - | ✅ | ✅ | ✅ |
| Headers/Footers | - | ✅ | ✅ | - |
| Embedded images | - | ✅ | ✅ | ✅ |

## Environment Setup

```bash
# Initialize virtual environment
./scripts/init-runtime.sh

# Activate
source scripts/.venv/bin/activate
```

## Dependencies

- Python 3.10+
- PaddleOCR / PPStructureV3
- python-docx, python-pptx, PyMuPDF, Pillow
