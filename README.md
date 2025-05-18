# Invoice Processor 🔍🧾  
Ein Python-Skript zum automatisierten Aufbereiten, Analysieren und Ablegen von Rechnungen (PDF & Bilder).  
Mehrseitige PDFs werden aufgetrennt, per OCR ausgelesen, nach Lieferant*innen erkannt und anschließend sinnvoll umbenannt und in fertige bzw. unsichere Ordner verschoben.

---

## Features

| Feature | Erklärung |
|---------|-----------|
| **Mehrseitige PDF-Erkennung** | Erkennt automatisch, ob eine Datei mehrere Rechnungen enthält und trennt diese bei Bedarf. |
| **OCR-Unterstützung (Tesseract)** | Nutzt zuerst den eingebetteten Textlayer; fällt dann auf Tesseract-OCR (Deutsch) zurück. |
| **Ähnlichkeits-Analyse** | Header, Footer, Ränder & Text-Content werden gewichtet, um Seiten derselben Rechnung zusammenzuführen (`header_similarity`, `footer_similarity`, …). |
| **Lieferanten-Erkennung** | Verwendet GPT-4 (oder das in `config.yaml` angegebene Modell), um Firmenname, Rechnungsnummer, Datum & Betrag zu extrahieren. |
| **Intelligentes Dateinamen-Schema** |  `YYYY-MM-DD_<Firma>_<RechnungsNr>_<Betrag>.pdf`  – fehlende Infos werden mit Platzhaltern ersetzt. |
| **Konfigurierbare Ordnerstruktur** | Standard: `Rechnungen/`, `Rechnungen-Unsicher/`, `Backup/`, `temp_invoices/` (alles anpassbar). |
| **CLI-Schalter** | Aggressives/konservatives Splitten, Schwellenwerte, Bundle-Modus u.v.m. |
| **GUI-Fallback** | Bei kritischen Fehlern erscheint ein kleines Tkinter-Popup. |

---

## Voraussetzungen

| Software | Version / Hinweis |
|----------|------------------|
| Python | ≥ 3.8 |
| [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) | inkl. Sprachpaket **`deu`** |
| [Poppler](https://poppler.freedesktop.org/) | Nur benötigt, wenn `pdf2image` nicht schon ein Poppler-Binary findet |

### Python-Pakete

Installiere die Abhängigkeiten am einfachsten via:

```bash
pip install -r requirements.txt
```

`requirements.txt` sollte mindestens enthalten:

```
pytesseract
pdf2image
PyPDF2
pillow
scikit-image
numpy
openai
PyYAML
```

---

## Installation

```bash
git clone https://github.com/<dein-user>/invoice-processor.git
cd invoice-processor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

*Optional*: Hinterlege globale Tools  
```bash
# macOS
brew install tesseract poppler

# Debian/Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-deu poppler-utils
```

---

## Konfiguration (`config.yaml`)

Beim ersten Start wird automatisch eine **`config.yaml`** mit Default-Werten erzeugt.  
Wichtige Felder:

```yaml
global:
  openai_key: "sk-..."        # kann auch über ENV gesetzt werden
  model: "gpt-4.1-mini"
  threshold: 0.5              # Default-Schwelle für Seitensimilarität
  aggressive_split: false
  processing_mode: "BUNDLE"   # oder "SINGLE"
  blank_page_threshold: 0.90

  directories:
    complete: "Rechnungen"
    incomplete: "Rechnungen-Unsicher"
    backup: "Backup"
    temp: "temp_invoices"

similarity_weights:
  header_similarity:   0.2
  footer_similarity:   0.1
  margin_similarity:   0.1
  text_similarity:     0.6

company_names:
  - "Vattenfall Europe Sales GmbH"
  - "Telekom Deutschland GmbH"
  # …
```

---

## Umweltvariablen

| Variable | Zweck |
|----------|-------|
| `OPENAI_API_KEY`            | wird verwendet, falls im YAML kein Key liegt |
| `AGGRESSIVE_INVOICE_SPLIT`  | `"true"` / `"false"` – überschreibt YAML |
| `INVOICE_SIMILARITY_THRESHOLD` | Zahl zwischen 0 & 1 – überschreibt YAML |
| `TESSDATA_PREFIX`           | Tesseract-Pfad, falls nicht systemweit verfügbar |

---

## Verwendung

Wechsle in den Ordner, in dem sich die zu verarbeitenden PDFs/Bilder befinden – oder übergib einen Pfad per `--input`.

```bash
python invoice.py [Optionen]
```

### Häufige Optionen

| Schalter | Beschreibung |
|----------|--------------|
| `--aggressive-split` | Jede Seite wird als eigene Rechnung behandelt (ignoriert Heuristiken). |
| `--conservative-split` | (Default) Belässt zusammengehörige Seiten. |
| `--threshold 0.7` | Setzt manuellen Ähnlichkeits-Schwellwert (0–1). |
| `--pdf-as-invoice` | Jede PDF-Datei wird **komplett** als eine Rechnung genommen (kein Seiten-Split). |
| `--bundle-mode` | Ermöglicht mehrere Rechnungen **innerhalb** einer PDF (Multirechnungs-PDF). |

Beispiel:

```bash
# Aus dem Ordner mit den PDFs
python invoice.py --aggressive-split --threshold 0.4
```

Ergebnis:

```
.
├── Backup/
│   └── <Originaldateien>.pdf
├── Rechnungen/
│   └── 2025-03-03_Telekom_4711_199EUR.pdf
└── Rechnungen-Unsicher/
    └── 2025-01-15_Unknown_NoNum_NoAmount.pdf
```

---

## Logging & Fehlersuche

* Standard-Level: `INFO`  
* Logmeldungen werden auf STDOUT ausgegeben; redirecte bei Bedarf:  
  `python invoice.py > invoice.log 2>&1`
* Bei fehlender Tesseract-Installation oder Poppler-Pfaden bekommst du einen Hinweis sowie einen Tkinter-Dialog.

---

## Beitrag leisten

1. Forke das Repo und erstelle einen Feature-Branch:  
   `git checkout -b feature/neues-ding`
2. Commit & Push:  
   `git commit -m "Füge cooles Ding hinzu"`  
   `git push origin feature/neues-ding`
3. Öffne einen Pull-Request 🚀

---

## Lizenz

Dieses Projekt steht unter der **MIT-Lizenz** – siehe [`LICENSE`](LICENSE).

---

## Autor

Tom Snubbel – [@zeroses](mailto:zeroses@hotmail.com)  
Für Fragen oder Bug-Reports einfach ein Issue anlegen!
